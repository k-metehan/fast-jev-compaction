import { JevClient, type JevClientOptions } from './client.js';
import { compact } from './compact.js';
import type { CompactOptions, CompactResult, Message } from './types.js';

export type CompactMessagesOptions = CompactOptions & JevClientOptions;

/**
 * `compact` with a `JevClient` built from the options (key from
 * `TYPESAFE_API_KEY` by default), with timers for the retry wait and the deadline.
 */
export function compactMessages(
  messages: readonly Message[],
  options: CompactMessagesOptions = {},
): Promise<CompactResult> {
  const after = (ms: number, fn: () => void) => {
    const timer = setTimeout(fn, ms);
    return { cancel: () => clearTimeout(timer) };
  };
  return compact(messages, new JevClient(options), { after, ...options });
}
