import { isRetryable, noulAnswer } from './request.js';
import { collectToolCalls, estimateTokens, fitState, imageCount } from './state.js';
import type {
  AttachedContent,
  CallAction,
  CallAnswer,
  CallDecision,
  CompactOptions,
  CompactResult,
  CompactionState,
  JevAsker,
  JevQuestions,
  Message,
  ResolvedCompactOptions,
  ToolCall,
} from './types.js';

export const DEFAULT_OPTIONS: ResolvedCompactOptions = {
  goal: '',
  instructions: '',
  keepThreshold: 0.5,
  preserveRecentMessages: 6,
  maxStateTokens: 25_000,
  maxRequestTokens: 30_000,
  truncateHeadChars: 300,
  attached: [],
};

/**
 * What one image in a tool result counts for in the size math, in characters:
 * about the 1,600 tokens Claude reads a full-size image as.
 */
export const IMAGE_CHARS = 6_000;

/** Jev requests in flight at once. */
export const MAX_CONCURRENT_REQUESTS = 4;

/** The wait before a failed request is tried again, when `sleep` is given. */
export const RETRY_DELAY_MS = 1_000;

/** Tokens the request envelope (`model`, key names) adds around state and questions. */
const REQUEST_OVERHEAD_TOKENS = 20;

function finite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function resolveOptions(options: CompactOptions = {}): ResolvedCompactOptions {
  return {
    goal: options.goal ?? DEFAULT_OPTIONS.goal,
    instructions: options.instructions?.trim() ?? DEFAULT_OPTIONS.instructions,
    keepThreshold: finite(options.keepThreshold, DEFAULT_OPTIONS.keepThreshold),
    preserveRecentMessages: Math.max(
      0,
      Math.floor(
        finite(options.preserveRecentMessages, DEFAULT_OPTIONS.preserveRecentMessages),
      ),
    ),
    maxStateTokens: Math.max(1, finite(options.maxStateTokens, DEFAULT_OPTIONS.maxStateTokens)),
    maxRequestTokens: Math.max(
      1,
      finite(options.maxRequestTokens, DEFAULT_OPTIONS.maxRequestTokens),
    ),
    truncateHeadChars: Math.max(
      0,
      Math.floor(finite(options.truncateHeadChars, DEFAULT_OPTIONS.truncateHeadChars)),
    ),
    attached: options.attached ?? DEFAULT_OPTIONS.attached,
  };
}

/** The two `noul` questions asked about one call: keep the call, keep its result. */
export function questionsFor(call: ToolCall): JevQuestions {
  return {
    [`call_${call.id}`]: {
      type: 'noul',
      instructions: `Tool call ${call.id} (${call.tool}) should stay in the history: knowing this call was made, with its input, still matters for what the assistant does next`,
    },
    [`result_${call.id}`]: {
      type: 'noul',
      instructions: `The full output of tool call ${call.id} (${call.tool}, ${call.resultChars} chars${
        call.resultImages ? ` and ${imageCount(call.resultImages)}` : ''
      }) should stay in the history verbatim: the assistant still needs its contents and re-running the tool would not do`,
    },
  };
}

/**
 * Splits the candidate calls into batches whose questions, together with the
 * (always complete) state, fit one request.
 */
export function batchCalls(
  calls: readonly ToolCall[],
  stateTokens: number,
  options: Pick<ResolvedCompactOptions, 'maxRequestTokens'>,
): ToolCall[][] {
  const budget = options.maxRequestTokens - stateTokens - REQUEST_OVERHEAD_TOKENS;
  const batches: ToolCall[][] = [];
  let current: ToolCall[] = [];
  let currentTokens = 0;
  for (const call of calls) {
    const tokens = estimateTokens(JSON.stringify(questionsFor(call)));
    if (current.length > 0 && currentTokens + tokens > budget) {
      batches.push(current);
      current = [];
      currentTokens = 0;
    }
    if (current.length === 0 && tokens > budget) {
      throw new Error(
        `state leaves no room for questions (~${stateTokens} of ${options.maxRequestTokens} tokens)`,
      );
    }
    current.push(call);
    currentTokens += tokens;
  }
  if (current.length > 0) batches.push(current);
  return batches;
}

export function decideCall(
  call: Pick<ToolCall, 'id' | 'tool' | 'pinned' | 'protected'>,
  answer: CallAnswer,
  options: Pick<ResolvedCompactOptions, 'keepThreshold'>,
): CallDecision {
  const base = { id: call.id, tool: call.tool, ...answer };
  if (call.pinned) return { ...base, action: 'keep', reason: 'pinned' };
  if (call.protected) return { ...base, action: 'keep', reason: 'protected' };
  if (answer.keepResult >= options.keepThreshold) {
    return { ...base, action: 'keep', reason: 'kept' };
  }
  if (answer.keepCall >= options.keepThreshold) {
    return { ...base, action: 'drop_result', reason: 'result_dropped' };
  }
  return { ...base, action: 'drop_call', reason: 'call_dropped' };
}

async function askBatch(
  asker: JevAsker,
  state: CompactionState,
  batch: readonly ToolCall[],
): Promise<Map<string, CallAnswer>> {
  const questions: JevQuestions = Object.assign({}, ...batch.map(questionsFor));
  const { answers } = await asker.ask(state, questions);
  return new Map(
    batch.map((call) => [
      call.id,
      {
        keepCall: noulAnswer(answers, `call_${call.id}`),
        keepResult: noulAnswer(answers, `result_${call.id}`),
      },
    ]),
  );
}

/** `askBatch`, tried once more when the failure is worth it (isRetryable). */
async function askWithRetry(
  asker: JevAsker,
  state: CompactionState,
  batch: readonly ToolCall[],
  sleep: ((ms: number) => Promise<void>) | undefined,
): Promise<Map<string, CallAnswer>> {
  try {
    return await askBatch(asker, state, batch);
  } catch (error) {
    if (!isRetryable(error)) throw error;
    await sleep?.(RETRY_DELAY_MS);
    return askBatch(asker, state, batch);
  }
}

/** Runs `run` over `items` with at most `limit` in flight; settles each, in order. */
async function settleLimited<T, R>(
  items: readonly T[],
  limit: number,
  run: (item: T) => Promise<R>,
): Promise<PromiseSettledResult<R>[]> {
  const settled: PromiseSettledResult<R>[] = new Array(items.length);
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < items.length) {
      const index = next;
      next += 1;
      try {
        settled[index] = { status: 'fulfilled', value: await run(items[index]!) };
      } catch (reason) {
        settled[index] = { status: 'rejected', reason };
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return settled;
}

function truncatedResultText(text: string, isError: boolean, headChars: number): string {
  if (text.length <= headChars + 120) return text;
  const head = headChars > 0 ? `${text.slice(0, headChars)}\n` : '';
  return `${head}[fast-jev-compaction truncated ${text.length - headChars} chars of this tool result${
    isError ? ' (error)' : ''
  }; re-run the tool if needed]`;
}

/**
 * One message under the actions: the same object when nothing in it changes,
 * a rebuilt copy, or null when it loses all its content. A dropped call goes
 * with its tool_use and its result; a dropped result keeps a bounded head and
 * note. The tool_use of a dropped result stays the same object: what the model
 * reads of a call's outcome is the tool_result alone.
 */
function rewriteMessage(
  message: Message,
  actions: ReadonlyMap<string, CallAction>,
  headChars: number,
): Message | null {
  const results = message.toolResults ?? [];
  const touched =
    message.toolUses.some((tool) => actions.get(tool.tool_use_id) === 'drop_call') ||
    results.some((result) => actions.has(result.tool_use_id));
  if (!touched) return message;
  const toolUses = message.toolUses.filter(
    (tool) => actions.get(tool.tool_use_id) !== 'drop_call',
  );
  const toolResults = results
    .filter((result) => actions.get(result.tool_use_id) !== 'drop_call')
    .map((result) => {
      if (actions.get(result.tool_use_id) !== 'drop_result') return result;
      const text = truncatedResultText(result.text, result.isError ?? false, headChars);
      const images = imageCount(result.images);
      if (text === result.text && !images) return result;
      // A dropped result loses its images whatever its length; the note says so.
      const note = images
        ? `[fast-jev-compaction removed ${images} from this tool result; re-run the tool if needed]`
        : '';
      return {
        tool_use_id: result.tool_use_id,
        text: [text, note].filter((part) => part.length > 0).join('\n'),
        isError: result.isError,
      };
    });
  if (
    toolUses.length === message.toolUses.length &&
    toolResults.length === results.length &&
    toolResults.every((result, index) => result === results[index])
  ) {
    return message;
  }
  if (message.text.trim().length === 0 && toolUses.length === 0 && toolResults.length === 0) {
    return null;
  }
  const rebuilt: Message = { role: message.role, text: message.text, toolUses };
  if (toolResults.length > 0) rebuilt.toolResults = toolResults;
  return rebuilt;
}

function actionsOf(
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
): Map<string, CallAction> {
  const byId = new Map(calls.map((call) => [call.id, call]));
  const actions = new Map<string, CallAction>();
  for (const decision of decisions) {
    const call = byId.get(decision.id);
    if (call && decision.action !== 'keep') actions.set(call.tool_use_id, decision.action);
  }
  return actions;
}

export type Rebuilt = {
  messages: Message[];
  /** The decisions as applied: a call whose attached content could not be put back is `protected`. */
  decisions: CallDecision[];
  /** The attached contents put back as text. */
  carried: AttachedContent[];
};

/**
 * Rebuilds the conversation from the decisions without losing attached
 * content (see AttachedContent). Untouched messages are returned as the same
 * objects they came in as; messages that lose all their content are removed.
 *
 * When every message holding an attached content's results is rebuilt or
 * removed, its text is put back after them: appended to the last one, or as a
 * user message of its own in its place. When only some are (the content's
 * owner is unknown), or the content has no text, those calls are kept instead.
 */
export function rebuild(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  headChars: number,
  attached: readonly AttachedContent[] = [],
): Rebuilt {
  const actions = actionsOf(decisions, calls);
  const holders = attached.map((content) => {
    const ids = new Set(content.toolUseIds);
    const indices: number[] = [];
    messages.forEach((message, index) => {
      if ((message.toolResults ?? []).some((result) => ids.has(result.tool_use_id))) indices.push(index);
    });
    return indices;
  });

  let out = messages.map((message) => rewriteMessage(message, actions, headChars));
  const lost = (index: number): boolean => out[index] !== messages[index];
  const reverted = new Set<string>();
  for (let pass = 0; pass < attached.length + 1; pass += 1) {
    let changed = false;
    attached.forEach((content, n) => {
      const indices = holders[n] ?? [];
      const gone = indices.filter(lost);
      if (gone.length === 0) return;
      if (content.text !== undefined && gone.length === indices.length) return;
      for (const index of gone) {
        for (const result of messages[index]?.toolResults ?? []) {
          if (actions.delete(result.tool_use_id)) {
            reverted.add(result.tool_use_id);
            changed = true;
          }
        }
      }
    });
    if (!changed) break;
    out = messages.map((message) => rewriteMessage(message, actions, headChars));
  }

  const putBack = new Map<number, string[]>();
  const carried: AttachedContent[] = [];
  attached.forEach((content, n) => {
    const indices = holders[n] ?? [];
    const text = content.text?.trim();
    if (!text || indices.length === 0 || !indices.every(lost)) return;
    const last = indices[indices.length - 1]!;
    putBack.set(last, [...(putBack.get(last) ?? []), text]);
    carried.push(content);
  });

  const kept: Message[] = [];
  out.forEach((message, index) => {
    const texts = putBack.get(index);
    if (!texts) {
      if (message) kept.push(message);
      return;
    }
    if (message) {
      kept.push({ ...message, text: [message.text, ...texts].filter((t) => t.length > 0).join('\n\n') });
    } else {
      kept.push({ role: 'user', text: texts.join('\n\n'), toolUses: [] });
    }
  });

  const byId = new Map(calls.map((call) => [call.id, call]));
  return {
    messages: kept,
    decisions: decisions.map((decision): CallDecision => {
      const call = byId.get(decision.id);
      return call && reverted.has(call.tool_use_id)
        ? { ...decision, action: 'keep', reason: 'protected' }
        : decision;
    }),
    carried,
  };
}

/** `rebuild`, returning the messages alone. */
export function applyDecisions(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  headChars: number,
  attached: readonly AttachedContent[] = [],
): Message[] {
  return rebuild(messages, decisions, calls, headChars, attached).messages;
}

/** Characters of text, tool input and tool output a message holds; an image counts IMAGE_CHARS. */
export function messageChars(message: Message): number {
  let total = message.text.length;
  for (const tool of message.toolUses) {
    try {
      total += JSON.stringify(tool.input).length;
    } catch {
      total += 20;
    }
  }
  for (const result of message.toolResults ?? []) {
    total += result.text.length + IMAGE_CHARS * Math.max(0, result.images ?? 0);
  }
  return total;
}

export function reductionRatio(result: Pick<CompactResult, 'stats'>): number {
  const { charsBefore, charsAfter } = result.stats;
  return charsBefore === 0 ? 0 : (charsBefore - charsAfter) / charsBefore;
}

function count(decisions: readonly CallDecision[], reason: CallDecision['reason']): number {
  return decisions.filter((decision) => decision.reason === reason).length;
}

/**
 * Compacts a transcript by asking Jev, for every tool call outside the pinned
 * first and newest messages, whether the call and whether its result must
 * stay. The whole history (results omitted, fitted into `maxStateTokens`) is
 * sent as state with every batch of questions. Content the host keeps beside
 * tool results (`attached`) survives: put back as text, or its calls kept.
 * Requests run at most MAX_CONCURRENT_REQUESTS at a time; one that is rate
 * limited, hits a server error or gets no answer is tried once more. The calls
 * of a request that still fails are kept. Throws when no request is answered,
 * Jev answers malformed, or the history cannot be fitted; the caller decides
 * whether to fall back.
 */
export async function compact(
  messages: readonly Message[],
  asker: JevAsker,
  options: CompactOptions = {},
): Promise<CompactResult> {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const held = new Set(
    resolved.attached.flatMap((content) => (content.text === undefined ? content.toolUseIds : [])),
  );
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages).map((call) =>
    held.has(call.tool_use_id) ? { ...call, protected: true } : call,
  );
  const candidates = calls.filter((call) => !call.pinned && !call.protected);
  const attachedChars = (contents: readonly AttachedContent[]): number =>
    contents.reduce((sum, content) => sum + (content.text?.trim().length ?? 0), 0);
  const charsBefore =
    messages.reduce((sum, message) => sum + messageChars(message), 0) +
    attachedChars(resolved.attached);

  let fitted: { tokens: number; stage: string } = { tokens: 0, stage: '' };
  let batches: ToolCall[][] = [];
  const answers = new Map<string, CallAnswer>();
  const failures: unknown[] = [];
  if (candidates.length > 0) {
    const state = fitState(messages, calls, resolved);
    fitted = state;
    batches = batchCalls(candidates, state.tokens, resolved);
    const settled = await settleLimited(batches, MAX_CONCURRENT_REQUESTS, (batch) =>
      askWithRetry(asker, state.state, batch, options.sleep),
    );
    for (const outcome of settled) {
      if (outcome.status === 'rejected') failures.push(outcome.reason);
      else for (const [id, answer] of outcome.value) answers.set(id, answer);
    }
    // Nothing answered: let the caller fall back with the reason.
    if (failures.length === batches.length) throw failures[0];
  }

  // A call Jev did not answer (its request failed) is kept.
  const decided = calls.map((call) =>
    decideCall(call, answers.get(call.id) ?? { keepCall: 1, keepResult: 1 }, resolved),
  );
  const { messages: kept, decisions, carried } = rebuild(
    messages,
    decided,
    calls,
    resolved.truncateHeadChars,
    resolved.attached,
  );
  return {
    messages: kept,
    decisions,
    stats: {
      messagesBefore: messages.length,
      messagesAfter: kept.length,
      charsBefore,
      // Attached content put back is in the messages now; the rest stays attached.
      charsAfter:
        kept.reduce((sum, message) => sum + messageChars(message), 0) +
        attachedChars(resolved.attached) -
        attachedChars(carried),
      calls: calls.length,
      kept: count(decisions, 'kept'),
      resultsDropped: count(decisions, 'result_dropped'),
      callsDropped: count(decisions, 'call_dropped'),
      pinned: count(decisions, 'pinned'),
      protected: count(decisions, 'protected'),
      carried: carried.length,
      stateTokens: fitted.tokens,
      stateStage: fitted.stage,
      requests: batches.length,
      failedRequests: failures.length,
      ...(failures.length > 0 && {
        requestError: failures[0] instanceof Error ? failures[0].message : String(failures[0]),
      }),
      ms: Date.now() - started,
    },
  };
}
