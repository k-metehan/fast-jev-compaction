import type {
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import { compact, reductionRatio, resolveOptions } from '../src/compact.js';
import { buildJevRequest, DEFAULT_MODEL, parseJevResponse } from '../src/request.js';
import type {
  AttachedContent,
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  compactAtPercent: 60,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
};

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  apiKey?: string;
  compactAtPercent: number;
  minReductionRatio: number;
  model: string;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxRequestTokens',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  return config;
}

/** A `JevAsker` over the engine's `$.http.fetch`. */
export function jevAsker(fetchFn: HookFetch, apiKey: string, model: string): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model }, state, questions);
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original. `libraryInput` is
 * what the library was given, index for index (copies of `input` where the
 * hook added what the rows do not show); it maps back onto `input`.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
  libraryInput: readonly Message[] = input,
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  input.forEach((message, index) => {
    const given = libraryInput[index] ?? message;
    messages.set(given, message);
    message.toolUses.forEach((tool, n) => uses.set(given.toolUses[n] ?? tool, tool));
    (message.toolResults ?? []).forEach((result, n) => results.set(given.toolResults?.[n] ?? result, result));
  });
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

/** One content block of the Messages API form (`$.session.messages({ as: 'api' })`). */
export type ApiBlock = { type: string; [field: string]: unknown };

/** One message of the Messages API form, as the next request is built from it. */
export type ApiMessage = { role: 'user' | 'assistant'; content: ApiBlock[] };

/**
 * Claude Code hands `session.compact` one row per user or assistant message
 * and keeps every other message of the transcript (attachments such as the
 * reminders and the messages the user typed while a tool ran, and meta user
 * messages such as skill bodies) beside the row before it. A row handed back
 * unchanged with its handle brings them back; a row rebuilt or left out loses
 * them, and the rows do not show them. The API form does: they are rendered
 * there into the user message that holds the tool results before them, folded
 * into a tool_result's text after the result itself, or as blocks of their own
 * after the results.
 *
 * Reads that content for each API user message holding tool results of the
 * rows, as AttachedContent: its text when it is text, no text when it is not
 * (an image) or cannot be told apart from the results (the calls are then
 * kept). A result the API form does not show is kept too.
 */
export function attachedContent(
  rows: readonly SessionMessage[],
  view: readonly ApiMessage[],
): AttachedContent[] {
  const own = new Map<string, string>();
  const texts = new Map<string, number>();
  for (const row of rows) {
    for (const result of row.toolResults ?? []) own.set(result.tool_use_id, result.text.trim());
    const text = row.role === 'user' ? row.text.trim() : '';
    if (text) texts.set(text, (texts.get(text) ?? 0) + 1);
  }
  const ownText = (text: string): boolean => {
    const left = texts.get(text) ?? 0;
    if (left === 0) return false;
    texts.set(text, left - 1);
    return true;
  };
  const textOf = (content: unknown): string =>
    typeof content === 'string'
      ? content
      : Array.isArray(content)
        ? content
            .filter((block): block is { type: 'text'; text: unknown } => block?.type === 'text')
            .map((block) => String(block.text ?? ''))
            .join('\n')
        : '';

  const attached: AttachedContent[] = [];
  const shown = new Set<string>();
  for (const message of view) {
    if (message.role !== 'user' || !Array.isArray(message.content)) continue;
    const ids = message.content.flatMap((block) =>
      block.type === 'tool_result' && typeof block['tool_use_id'] === 'string' && own.has(block['tool_use_id'])
        ? [block['tool_use_id']]
        : [],
    );
    if (ids.length === 0) continue;
    for (const id of ids) shown.add(id);
    const extra: string[] = [];
    let readable = true;
    for (const block of message.content) {
      if (block.type === 'tool_result') {
        const id = block['tool_use_id'];
        const mine = typeof id === 'string' ? own.get(id) : undefined;
        if (mine === undefined) {
          readable = false;
          continue;
        }
        const text = textOf(block['content']);
        if (text.trim() === mine) continue;
        if (text.startsWith(mine)) {
          const rest = text.slice(mine.length).trim();
          if (rest) extra.push(rest);
        } else readable = false;
      } else if (block.type === 'text') {
        const text = String(block['text'] ?? '').trim();
        if (text && !ownText(text)) extra.push(text);
      } else {
        readable = false;
      }
    }
    if (!readable) attached.push({ toolUseIds: ids });
    else if (extra.length > 0) attached.push({ toolUseIds: ids, text: extra.join('\n\n') });
  }
  for (const id of own.keys()) if (!shown.has(id)) attached.push({ toolUseIds: [id] });
  return attached;
}

/**
 * The image blocks inside each tool result, by tool_use_id. The rows hold a
 * result's text blocks only, so a screenshot is invisible there.
 */
export function resultImages(view: readonly ApiMessage[]): Map<string, number> {
  const images = new Map<string, number>();
  for (const message of view) {
    if (message.role !== 'user' || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      const id = block['tool_use_id'];
      const content = block['content'];
      if (block.type !== 'tool_result' || typeof id !== 'string' || !Array.isArray(content)) continue;
      const n = content.filter((part) => (part as { type?: unknown } | null)?.type === 'image').length;
      if (n > 0) images.set(id, n);
    }
  }
  return images;
}

type ApiViewSource = {
  session: { messages: (args: { as: 'api'; agentId?: string }) => Promise<unknown> };
};

/** The conversation being compacted in API form; throws when the host does not give it. */
export async function apiView($: ApiViewSource, agentId?: string): Promise<ApiMessage[]> {
  const view = await $.session.messages(agentId === undefined ? { as: 'api' } : { as: 'api', agentId });
  if (!Array.isArray(view)) {
    const deny = (view as { deny?: unknown } | null)?.deny;
    throw new Error(`no API view of the conversation${typeof deny === 'string' ? ` (${deny})` : ''}`);
  }
  if (!view.every((message) => Array.isArray((message as { content?: unknown })?.content))) {
    throw new Error('no API view of the conversation (this Claude Code answers rows only)');
  }
  return view as ApiMessage[];
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/**
 * Runs the library over a session transcript; throws when the key is missing
 * or Jev fails. `images` are the image blocks of tool results, by tool_use_id
 * (resultImages): the library is given copies of those rows that carry them,
 * and the engine's rows come back where nothing changed.
 */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
  images: ReadonlyMap<string, number> = new Map(),
): Promise<SessionCompaction> {
  if (!config.apiKey) throw new Error('TYPESAFE_API_KEY is not configured');
  const given: Message[] = messages.map((message) =>
    (message.toolResults ?? []).some((result) => images.has(result.tool_use_id))
      ? {
          ...message,
          toolResults: (message.toolResults ?? []).map((result) =>
            images.has(result.tool_use_id) ? { ...result, images: images.get(result.tool_use_id) } : result,
          ),
        }
      : message,
  );
  const result = await compact(given, jevAsker(fetchFn, config.apiKey, config.model), config);
  return { result, messages: toSessionMessages(messages, result.messages, given) };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
    stats.protected > 0 ? `${stats.protected} protected` : '',
    stats.carried > 0 ? `${stats.carried} attached put back` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned')
    .map(
      (d) =>
        `${d.id}:${d.tool}:${d.action}/call=${d.keepCall.toFixed(2)}/result=${d.keepResult.toFixed(2)}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

async function getApiKey(
  $: {
    env: { get: (name: string) => Promise<string | undefined> };
    settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
  },
  config: HookConfig,
): Promise<string | undefined> {
  if (config.apiKey) return config.apiKey;
  const fromEnv = await $.env.get('TYPESAFE_API_KEY');
  if (fromEnv) return fromEnv;
  const settings = await $.settings.read();
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)['TYPESAFE_API_KEY'];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

// Details go to the debug log; the user sees nothing but a short "compaction
// done" toast when Jev's compaction replaced the history. The debug log is
// written only under --debug (never in the desktop app), so each compaction's
// outcome also goes, as one line, to the plugin's own log file.
type DebugLog = { ui: { log: (text: string, options?: { to?: 'debug' }) => void } };

function debugLog($: DebugLog, text: string): void {
  $.ui.log(text, { to: 'debug' });
}

/** Lines the compaction log keeps; older ones are dropped. */
export const COMPACTION_LOG_LINES = 200;

type LogHost = DebugLog & {
  env: { get: (name: string) => Promise<string | undefined> };
  fs: {
    read: (path: string) => Promise<string>;
    write: (path: string, text: string) => Promise<void>;
  };
};

/**
 * The plugin's own log: `<config dir>/fast-jev-compaction/compactions.log`,
 * the config dir being `CLAUDE_CONFIG_DIR` or `~/.claude`. Claude Code gives
 * function hooks no plugin data directory (`$.plugin` is its name and root).
 */
export async function compactionLogPath($: Pick<LogHost, 'env'>): Promise<string | undefined> {
  const configDir = await $.env.get('CLAUDE_CONFIG_DIR');
  const home = await $.env.get('HOME');
  const base = configDir || (home ? `${home}/.claude` : undefined);
  return base ? `${base.replace(/\/+$/, '')}/fast-jev-compaction/compactions.log` : undefined;
}

/** Appends one timestamped line, keeping the newest COMPACTION_LOG_LINES; never throws. */
export async function appendCompactionLog($: LogHost, line: string, now: Date = new Date()): Promise<void> {
  try {
    const path = await compactionLogPath($);
    if (!path) return;
    let previous = '';
    try {
      previous = await $.fs.read(path);
    } catch {
      // No log yet.
    }
    const lines = previous.split('\n').filter((l) => l.length > 0);
    lines.push(`${now.toISOString()} ${line.replace(/\s*\n\s*/g, ' ')}`);
    await $.fs.write(path, `${lines.slice(-COMPACTION_LOG_LINES).join('\n')}\n`);
  } catch (error) {
    debugLog($, `compaction log not written (${error instanceof Error ? error.message : String(error)})`);
  }
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  let compacting = false;
  // Headless (-p / SDK, so the desktop app too) rejects $.session.compact on every
  // call; after the first rejection the host's own auto-compact is the only trigger.
  let headless = false;

  on('session.compact', async ($, event, next) => {
    const which = `${event.trigger}${event.agentId === undefined ? '' : ` (agent ${event.agentId})`}`;
    const report = async (text: string): Promise<void> => {
      debugLog($, text);
      await appendCompactionLog($, `${which} ${text}`);
    };
    try {
      const apiKey = await getApiKey($, configured);
      const view = await apiView($, event.agentId);
      // What Claude Code keeps beside the rows; a row rebuilt or left out would lose it.
      const attached = attachedContent(event.messages, view);
      const config = { ...configured, apiKey, attached };
      const fetchFn: HookFetch = async (url, init) => {
        const response = await $.http.fetch(url, init);
        return { status: response.status, ok: response.ok, text: response.text };
      };
      const { result, messages } = await compactSession(event.messages, config, fetchFn, resultImages(view));
      if (reductionRatio(result) < config.minReductionRatio) {
        await report(
          `fallback to built-in summary (below ${percent(config.minReductionRatio)} minimum: ${summarize(result)})`,
        );
        return next(event);
      }
      await report(`kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)})`);
      $.ui.toast('compaction done');
      return { messages };
    } catch (error) {
      await report(`fallback to built-in summary (${error instanceof Error ? error.message : String(error)})`);
      return next(event);
    }
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    if (compacting || headless) return next(event);
    try {
      const { context } = await $.session.usage();
      if ((context.percent ?? 0) < configured.compactAtPercent) return next(event);
      compacting = true;
      await $.session.compact();
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error);
      if (text.includes('headless')) headless = true;
      debugLog($, headless ? `auto-compact off for this session (${text})` : `auto-compact skipped (${text})`);
    } finally {
      compacting = false;
    }
    return next(event);
  });
};

export { resolveOptions };
