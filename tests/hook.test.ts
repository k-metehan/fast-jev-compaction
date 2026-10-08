import { describe, expect, it } from 'vitest';
import {
  COMPACTION_LOG_LINES,
  compactionLogPath,
  compactSession,
  decisionLog,
  decisionLogLines,
  register,
  resolveHookConfig,
  summarize,
  toSessionMessages,
} from '../hooks/fast-jev.ts';
import { applyDecisions, collectToolCalls, decideCall, type Message } from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('tool-2', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'Fixing now.', { handle: 'h-5' }),
    message('user', 'go ahead', { handle: 'h-6' }),
  ];
}

function jevFetch(answer: (name: string) => number, bodies: string[] = []) {
  return async (_url: string, init?: { body?: string }) => {
    bodies.push(init?.body ?? '');
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

describe('hook config', () => {
  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfig({})).toEqual({ compactAtPercent: 60, minReductionRatio: 0.25, model: 'jev-latest' });
    expect(
      resolveHookConfig({ apiKey: 'k', keepThreshold: 0.3, maxStateTokens: 1000, model: 'jev-x', goal: 'g', compactAtPercent: 'no' }),
    ).toEqual({
      apiKey: 'k',
      keepThreshold: 0.3,
      maxStateTokens: 1000,
      model: 'jev-x',
      goal: 'g',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
    });
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    messages[1]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[2]!.toolResults![0]!.text = 'x'.repeat(2000);
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out).toHaveLength(messages.length);
    expect(out[0]).toBe(messages[0]);
    // A dropped result leaves its call alone, so the call keeps its handle (and the
    // hidden messages the engine keeps beside it).
    expect(out[1]).toBe(messages[1]);
    expect(out[1]?.handle).toBe('h-tool-1');
    expect(out[2]?.handle).toBeUndefined();
    expect(out[2]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'tool-1', isError: false });
    expect(out[3]).toBe(messages[3]);
    expect(out[4]).toBe(messages[4]);
  });

  it('preserves short dropped-result messages and their handles', () => {
    const messages = transcript();
    messages[1]!.toolUses[0]!.text = 'y'.repeat(100);
    messages[2]!.toolResults![0]!.text = 'y'.repeat(100);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out[1]).toBe(messages[1]);
    expect(out[2]).toBe(messages[2]);
  });
});

describe('compactSession', () => {
  it('runs the library over the engine fetch and reports the outcome', async () => {
    const bodies: string[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k', model: 'jev-x' };
    const { result: output, messages } = await compactSession(
      transcript(),
      config,
      jevFetch((name) => (name === 'call_t2' || name === 'result_t2' ? 0.9 : 0.1), bodies),
    );
    expect(bodies).toHaveLength(1);
    expect(JSON.parse(bodies[0]!).model).toBe('jev-x');
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_call', 'keep']);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', 'h-tool-2', 'r-tool-2', 'h-5', 'h-6']);
    expect(summarize(output)).toMatch(/^\d+% reduction; 1 kept, 1 call_dropped; state ~\d+ tokens \(full\) in 1 request\(s\)$/);
    expect(decisionLog(output)).toBe('t1:Read:drop_call/call=0.10/result=0.10 t2:Bash:keep/call=0.90/result=0.90');
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const { result: output } = await compactSession(transcript(), config, jevFetch(() => 0.1));
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([
      'decisions (1/2): t1:Read:drop_call/call=0.10/result=0.10',
      'decisions (2/2): t2:Bash:drop_call/call=0.10/result=0.10',
    ]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('throws on a missing key and on failed requests so the hook falls back', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    await expect(compactSession(transcript(), config, jevFetch(() => 0))).rejects.toThrow(/TYPESAFE_API_KEY/);
    await expect(
      compactSession(transcript(), { ...config, apiKey: 'k' }, async () => ({ status: 500, ok: false, text: 'x' })),
    ).rejects.toThrow(/500/);
  });
});

describe('session.compact hook', () => {
  const LOG = '/home/me/.claude/fast-jev-compaction/compactions.log';

  function run(
    fetch: (url: string, init?: { body?: string }) => Promise<{ status: number; ok: boolean; text: string }>,
    files: Map<string, string> = new Map(),
    event: { instructions?: string } = {},
  ) {
    const handlers: Record<string, Function> = {};
    register(((name: string, h: Function) => { handlers[name] = h; }) as never, { preserveRecentMessages: 1 } as never);
    const logs: { text: string; to?: string }[] = [];
    const toasts: string[] = [];
    // The API form of transcript(): nothing kept beside its rows.
    const view = transcript().map((row) => ({
      role: row.role,
      content: [
        ...(row.toolResults ?? []).map((r) => ({ type: 'tool_result', tool_use_id: r.tool_use_id, content: r.text })),
        ...(row.text ? [{ type: 'text', text: row.text }] : []),
        ...row.toolUses.map((u) => ({ type: 'tool_use', id: u.tool_use_id, name: u.tool, input: u.input })),
      ],
    }));
    const env: Record<string, string> = { TYPESAFE_API_KEY: 'k', HOME: '/home/me' };
    const sleeps: number[] = [];
    const $ = {
      env: { get: async (name: string) => env[name] },
      settings: { read: async () => ({}) },
      session: { messages: async () => view },
      clock: {
        // Short waits fire at once; the deadline never does.
        after: (ms: number, fn: () => void) => {
          sleeps.push(ms);
          if (ms < 10_000) queueMicrotask(fn);
          return { cancel: () => undefined };
        },
      },
      http: { fetch: async (url: string, init?: { body?: string }) => fetch(url, init) },
      fs: {
        read: async (path: string) => {
          const text = files.get(path);
          if (text === undefined) throw new Error('ENOENT');
          return text;
        },
        write: async (path: string, text: string) => {
          files.set(path, text);
        },
      },
      ui: {
        log: (text: string, options?: { to?: string }) => logs.push({ text, to: options?.to }),
        toast: (text: string) => toasts.push(text),
      },
    };
    return handlers['session.compact']($, { trigger: 'manual', messages: transcript(), ...event }, async () => ({})).then(() => ({
      logs,
      toasts,
      sleeps,
      lines: (files.get(LOG) ?? '').split('\n').filter(Boolean),
    }));
  }

  it('shows only "compaction done" and logs the outcome to the debug log and its own log', async () => {
    const { logs, toasts, lines } = await run(jevFetch(() => 0.1));
    expect(toasts).toEqual(['compaction done']);
    expect(logs).toHaveLength(1);
    expect(logs[0]).toEqual({ text: expect.stringMatching(/^kept/), to: 'debug' });
    expect(lines).toEqual([expect.stringMatching(/^\d{4}-\d\d-\d\dT[\d:.]+Z manual kept \d+\/7 messages, no summary \(/)]);
  });

  it('shows nothing on a fallback and logs its reason to the debug log and its own log', async () => {
    const { logs, toasts, lines } = await run(async () => ({ status: 500, ok: false, text: 'x' }));
    expect(toasts).toEqual([]);
    expect(logs).toHaveLength(1);
    expect(logs[0]).toEqual({ text: expect.stringMatching(/^fallback/), to: 'debug' });
    expect(lines).toEqual([expect.stringMatching(/Z manual fallback to built-in summary \(Jev request failed \(500\): x\)$/)]);
  });

  it('hands Jev the text after /compact as part of the goal', async () => {
    const bodies: string[] = [];
    await run(jevFetch(() => 0.1, bodies), new Map(), { instructions: 'keep the npm test output' });
    expect(JSON.parse(bodies[0]!).state.goal).toMatch(/\nThe user asked this compaction to keep: keep the npm test output$/);
  });

  it('tries a failed Jev request once more, after a second on a host timer, under a 45 s deadline', async () => {
    let calls = 0;
    const { sleeps, lines } = await run(async () => {
      calls += 1;
      return { status: 503, ok: false, text: 'busy' };
    });
    expect(calls).toBe(2);
    expect(sleeps).toEqual([45_000, 1000]);
    expect(lines[0]).toMatch(/fallback to built-in summary \(Jev request failed \(503\): busy\)$/);
  });

  it('keeps its own log to the newest lines, one per compaction', async () => {
    const files = new Map([[LOG, Array.from({ length: COMPACTION_LOG_LINES }, (_, i) => `old ${i}`).join('\n')]]);
    const { lines } = await run(jevFetch(() => 0.1), files);
    expect(lines).toHaveLength(COMPACTION_LOG_LINES);
    expect(lines[0]).toBe('old 1');
    expect(lines.at(-1)).toMatch(/ manual kept /);
  });

  it('puts its log under CLAUDE_CONFIG_DIR when set', async () => {
    const at = async (env: Record<string, string>) => compactionLogPath({ env: { get: async (name: string) => env[name] } });
    expect(await at({ HOME: '/home/me' })).toBe(LOG);
    expect(await at({ HOME: '/home/me', CLAUDE_CONFIG_DIR: '/cfg/' })).toBe('/cfg/fast-jev-compaction/compactions.log');
    expect(await at({})).toBeUndefined();
  });
});
