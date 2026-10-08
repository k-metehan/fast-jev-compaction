export type Role = 'user' | 'assistant';

/**
 * A tool_use block of an assistant message. `text` and `isError` mirror the
 * outcome once the transcript holds it (Claude Code attaches them).
 */
export interface ToolUse {
  tool_use_id: string;
  tool: string;
  input: Record<string, unknown>;
  text?: string;
  isError?: boolean;
}

/** A tool_result block of a user message. */
export interface ToolResult {
  tool_use_id: string;
  /** Its text blocks; images are not in it. */
  text: string;
  isError?: boolean;
  /** Image blocks it carries besides its text (a screenshot). Default 0. */
  images?: number;
}

/**
 * One transcript message. The shape is a subset of Claude Code's
 * `SessionMessage`, so a session transcript can be passed in as is.
 */
export interface Message {
  role: Role;
  text: string;
  toolUses: ToolUse[];
  toolResults?: ToolResult[];
}

/** A tool call paired with its result by `tool_use_id`. */
export interface ToolCall {
  /** Short id used in the Jev state and question names (`t1`, `t2`, ...). */
  id: string;
  tool_use_id: string;
  tool: string;
  input: Record<string, unknown>;
  /** Index of the message holding the tool_use block. */
  callIndex: number;
  /** Index of the message holding the tool_result block. */
  resultIndex: number;
  resultChars: number;
  /** Image blocks the result carries besides its text. */
  resultImages?: number;
  isError: boolean;
  /** In the first or the newest preserved messages; never a candidate. */
  pinned: boolean;
  /** Its result carries attached content that cannot be put back; never a candidate. */
  protected?: boolean;
}

/**
 * Content the host keeps beside some tool results, outside the messages, and
 * loses when a message holding one of those results is rebuilt or removed. In
 * Claude Code: the reminders, the messages the user typed while a tool ran,
 * and skill bodies that follow a tool result.
 */
export interface AttachedContent {
  /**
   * The calls it sits beside. The host may keep it beside any message from the
   * first holding one of their calls or results to the last, so it is lost
   * only when every one of those messages is rebuilt or removed.
   */
  toolUseIds: string[];
  /**
   * The content as text, put back after those messages when they are all
   * rebuilt or removed. Absent when it must not be put back as text (an image,
   * or content that is not the user's own words, or that could not be
   * attributed): those calls are then kept.
   */
  text?: string;
}

export interface CallAnswer {
  /** Jev's probability that the call itself still matters. */
  keepCall: number;
  /** Jev's probability that the full result still needs to stay verbatim. */
  keepResult: number;
}

export type CallAction = 'keep' | 'drop_result' | 'drop_call';

export interface CallDecision extends CallAnswer {
  id: string;
  tool: string;
  action: CallAction;
  /**
   * `protected`: kept because dropping it would lose attached content that
   * cannot be put back (see AttachedContent).
   */
  reason: 'pinned' | 'protected' | 'kept' | 'result_dropped' | 'call_dropped';
}

export interface HistoryToolCall {
  id: string;
  tool: string;
  input: string;
  result: string;
}

export interface HistoryEntry {
  i: number;
  role: Role;
  text: string;
  /** Structured per call, or one compact line per call once the state has to shrink. */
  tool_calls?: HistoryToolCall[] | string[];
}

/** The state sent with every Jev request: the whole history, results omitted. */
export interface CompactionState {
  context: string;
  goal: string;
  history: HistoryEntry[];
}

export interface FittedState {
  state: CompactionState;
  tokens: number;
  /** Which fitting stage produced the state, for diagnostics. */
  stage: string;
}

export interface CompactOptions {
  /** Ongoing task description; defaults to the last few user prompts. */
  goal?: string;
  /** What the user asked this compaction to keep (`/compact <instructions>`); added to the goal. */
  instructions?: string;
  /** Minimum keep probability for a call or result to stay. Default 0.5. */
  keepThreshold?: number;
  /**
   * Newest messages never touched (the first message is always kept). Default 6.
   * Consecutive entries of one role count as one message, as the API sends
   * them; in Claude Code 6 is the last three tool steps (call and result).
   */
  preserveRecentMessages?: number;
  /** Estimated token ceiling for the state. Default 25000. */
  maxStateTokens?: number;
  /** Estimated token ceiling for state plus one batch of questions. Default 30000. */
  maxRequestTokens?: number;
  /** Characters of a dropped tool result to retain. Default 300. */
  truncateHeadChars?: number;
  /** Content the host keeps beside tool results (see AttachedContent). Default none. */
  attached?: readonly AttachedContent[];
  /**
   * Waits before a Jev request is tried again (rate limited, a server error,
   * no answer). Default: try again at once.
   */
  sleep?: (ms: number) => Promise<void>;
}

export interface ResolvedCompactOptions {
  goal: string;
  instructions: string;
  keepThreshold: number;
  preserveRecentMessages: number;
  maxStateTokens: number;
  maxRequestTokens: number;
  truncateHeadChars: number;
  attached: readonly AttachedContent[];
}

export interface CompactResult {
  /** The compacted transcript; untouched messages are the input objects. */
  messages: Message[];
  decisions: CallDecision[];
  stats: {
    messagesBefore: number;
    messagesAfter: number;
    charsBefore: number;
    charsAfter: number;
    calls: number;
    kept: number;
    resultsDropped: number;
    callsDropped: number;
    pinned: number;
    /** Calls kept because their attached content could not be put back. */
    protected: number;
    /** Attached contents put back as text after their rebuilt or removed results. */
    carried: number;
    stateTokens: number;
    /** Which fitting stage the state needed, '' when no request was made. */
    stateStage: string;
    requests: number;
    /** Requests that failed even when tried again; their calls were kept. */
    failedRequests: number;
    /** Why the first of them failed. */
    requestError?: string;
    ms: number;
  };
}

/** The `state` of a Jev request: a string or any JSON-serialisable object. */
export type JevState = string | object;

export interface NoulQuestion {
  type: 'noul';
  instructions: string;
  criteria?: {
    true?: string;
    false?: string;
  };
}

export interface ChoiceQuestion {
  type: 'choice';
  instructions: string;
  criteria: Record<string, string | null>;
}

export interface ScoreQuestion {
  type: 'score';
  instructions: string;
  criteria: string[];
}

export type JevQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion;
export type JevQuestions = Record<string, JevQuestion>;

export interface NoulAnswer {
  type?: 'noul';
  noul: number;
}

export interface ChoiceAnswer {
  type?: 'choice';
  choice: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface ScoreAnswer {
  type?: 'score';
  score: number;
  confidence: number;
  probabilities: Record<string, number>;
}

export type JevAnswer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

export interface JevResponse {
  model?: string;
  answers: Record<string, JevAnswer>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
  [key: string]: unknown;
}

/** Anything that can answer Jev questions: `JevClient`, or a host-provided adapter. */
export interface JevAsker {
  ask(state: JevState, questions: JevQuestions): Promise<JevResponse>;
}
