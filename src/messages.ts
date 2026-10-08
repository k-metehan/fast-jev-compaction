import { JevClient, type JevClientOptions } from './client.js';
import { compact } from './compact.js';
import type { CompactOptions, CompactResult, Message } from './types.js';

export type CompactMessagesOptions = CompactOptions & JevClientOptions;

/**
 * `compact` with a `JevClient` built from the options (key from
 * `TYPESAFE_API_KEY` by default), waiting with a timer before a retry.
 */
export function compactMessages(
  messages: readonly Message[],
  options: CompactMessagesOptions = {},
): Promise<CompactResult> {
  const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
  return compact(messages, new JevClient(options), { sleep, ...options });
}
