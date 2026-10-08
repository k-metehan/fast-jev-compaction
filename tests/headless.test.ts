import { describe, expect, it } from 'vitest';
import { register } from '../hooks/fast-jev.ts';

describe('turn.complete in a headless session', () => {
  it('stops retrying after the host says compact is unavailable', async () => {
    const handlers: Record<string, Function> = {};
    register(((name: string, h: Function) => { handlers[name] = h; return { catch: () => undefined }; }) as never, {} as never);
    const logs: { text: string; to?: string }[] = [];
    let compacts = 0;
    const $ = {
      session: {
        usage: async () => ({ context: { percent: 70 } }),
        compact: async () => {
          compacts++;
          throw new Error('$.session.compact: not available in a headless (-p / SDK) session yet');
        },
      },
      ui: {
        log: (text: string, options?: { to?: string }) => logs.push({ text, to: options?.to }),
        toast: () => {
          throw new Error('no toast expected');
        },
      },
    };
    for (let i = 0; i < 3; i++) await handlers['turn.complete']($, {}, async () => undefined);
    expect(compacts).toBe(1);
    expect(logs).toEqual([{ text: expect.stringMatching(/^auto-compact off/), to: 'debug' }]);
  });
});
