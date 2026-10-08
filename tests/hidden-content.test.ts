import { describe, expect, it } from 'vitest';
import {
  attachedContent,
  compactSession,
  register,
  resolveHookConfig,
  type ApiMessage,
} from '../hooks/fast-jev.ts';
import { apiViewOf, host, transcriptBuilder, type Raw, type Row } from './host.ts';

const TYPED = 'ALSO DO NOT TOUCH config.json';
const SKILL = '# Deploy skill: deploy with care, run the smoke test first.';
const output = (step: number) => `line of build output for step ${step}\n`.repeat(60);

/**
 * Six tool steps; the user types a message while step 2 runs (a queued_command
 * attachment after its result), step 3 loads a skill (a meta user message
 * after its result), and every result is followed by a token reminder.
 */
function session(extra: { imageAfterStep2?: boolean } = {}) {
  const t = transcriptBuilder();
  const raw: Raw[] = [t.prompt('Fix the build')];
  const hidden: Raw[] = [];
  for (let step = 1; step <= 6; step += 1) {
    raw.push(t.thinking());
    if (step === 3) {
      raw.push(t.use('t3', 'Skill', { skill: 'deploy' }), t.result('t3', `Launching skill: deploy\n${output(3)}`));
      const body = t.skillBody(SKILL);
      raw.push(body);
      hidden.push(body);
    } else {
      raw.push(t.use(`t${step}`, 'Bash', { command: `step ${step}` }), t.result(`t${step}`, output(step)));
    }
    if (step === 2) {
      const typed = t.queued(TYPED);
      raw.push(typed);
      hidden.push(typed);
      if (extra.imageAfterStep2) raw.push(t.image());
    }
    raw.push(t.tokens(900_000 - step));
  }
  raw.push(t.say('done'));
  return { raw, hidden };
}

/** A Jev answering per question name, over the hook's fetch. */
function jev(answer: (name: string) => number) {
  return async (_url: string, init?: { body?: string }) => {
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

/** drop_result or drop_call for the named calls (Jev's ids: t1, t2, … in call order), keep for the rest. */
function answers(mode: 'drop_result' | 'drop_call', ids: string[]) {
  return (name: string) => {
    const id = name.replace(/^(call|result)_/, '');
    if (!ids.includes(id)) return 0.9;
    return mode === 'drop_result' && name.startsWith('call_') ? 0.9 : 0.1;
  };
}

const OPTIONS = { preserveRecentMessages: 2, minReductionRatio: 0 };

/** Runs the plugin's session.compact hook against the copy of the host. */
async function compactThroughHook(
  raw: Raw[],
  answer: (name: string) => number,
  view: (args: unknown) => unknown = () => apiViewOf(raw),
) {
  const engine = host();
  const rows = engine.rowsOf(raw);
  const handlers: Record<string, Function> = {};
  register(((name: string, hook: Function) => {
    handlers[name] = hook;
  }) as never, OPTIONS as never);
  const logs: string[] = [];
  const toasts: string[] = [];
  let fellBack = false;
  const $ = {
    env: { get: async () => 'k' },
    settings: { read: async () => ({}) },
    session: { messages: async (args: unknown) => view(args) },
    http: { fetch: jev(answer) },
    ui: { log: (text: string) => logs.push(text), toast: (text: string) => toasts.push(text) },
  };
  const out = (await handlers['session.compact']!($, { trigger: 'manual', messages: rows }, async () => {
    fellBack = true;
    return { skip: 'core' };
  })) as { messages?: Row[] };
  return { engine, rows, out, final: out.messages ? engine.messagesOf(out.messages) : [], logs, toasts, fellBack };
}

const count = (messages: unknown[], text: string) => JSON.stringify(messages).split(text).length - 1;
const indexOf = (messages: unknown[], text: string) =>
  messages.findIndex((message) => JSON.stringify(message).includes(text));

describe('hidden messages beside a rewritten tool result (Claude Code 2.1.292)', () => {
  it('reproduces the loss when compaction ignores them', async () => {
    const { raw } = session();
    for (const mode of ['drop_result', 'drop_call'] as const) {
      const engine = host();
      const rows = engine.rowsOf(raw);
      const config = { ...resolveHookConfig(OPTIONS), apiKey: 'k' };
      const { messages } = await compactSession(rows, config, jev(answers(mode, ['t2', 't3'])));
      const final = engine.messagesOf(messages as Row[]);
      expect(count(final, TYPED), mode).toBe(0);
      expect(count(final, SKILL), mode).toBe(0);
    }
  });

  for (const mode of ['drop_result', 'drop_call'] as const) {
    it(`keeps the typed message and the skill body under ${mode}`, async () => {
      const { raw } = session();
      const { out, final, toasts, logs } = await compactThroughHook(raw, answers(mode, ['t2', 't3']));
      expect(toasts).toEqual(['compaction done']);
      expect(out.messages).toBeDefined();
      expect(count(final, TYPED)).toBe(1);
      expect(count(final, SKILL)).toBe(1);
      expect(logs.join('\n')).toMatch(/attached put back/);

      // In place: after step 2's call, before step 3's; after step 3's, before step 4's.
      const typedAt = indexOf(final, TYPED);
      const skillAt = indexOf(final, SKILL);
      expect(typedAt).toBeGreaterThan(indexOf(final, '"step 1"'));
      expect(typedAt).toBeLessThan(skillAt);
      expect(skillAt).toBeLessThan(indexOf(final, '"step 4"'));
      if (mode === 'drop_result') {
        expect(count(final, 'fast-jev-compaction truncated')).toBe(2);
        expect(typedAt).toBeGreaterThan(indexOf(final, '"step 2"'));
      } else {
        expect(count(final, '"step 2"')).toBe(0);
        expect(count(final, 'Launching skill')).toBe(0);
      }
    });
  }

  it('keeps a call whose hidden content cannot be put back as text', async () => {
    const { raw, hidden } = session({ imageAfterStep2: true });
    const { out, final } = await compactThroughHook(raw, answers('drop_call', ['t2', 't3']));
    expect(out.messages).toBeDefined();
    // Step 2's result row came back with its handle: the host restores its group as is.
    expect(final).toContain(hidden[0]);
    expect(count(final, TYPED)).toBe(1);
    expect(count(final, SKILL)).toBe(1);
    expect(count(final, '"step 2"')).toBe(1);
  });

  it('keeps parallel results whole when Jev drops only some of them', async () => {
    const t = transcriptBuilder();
    const typed = t.queued(TYPED);
    const raw: Raw[] = [
      t.prompt('Fix the build'),
      t.thinking(),
      t.use('a', 'Bash', { command: 'step a' }),
      t.use('b', 'Bash', { command: 'step b' }),
      t.result('a', output(1)),
      t.result('b', output(2)),
      typed,
      t.tokens(1000),
      t.say('both done'),
      t.thinking(),
      t.use('c', 'Bash', { command: 'step c' }),
      t.result('c', output(3)),
      t.say('done'),
    ];
    const mixed = await compactThroughHook(raw, answers('drop_result', ['t1']));
    expect(mixed.final).toContain(typed);
    expect(count(mixed.final, 'fast-jev-compaction truncated')).toBe(0);
    expect(mixed.logs.join('\n')).toMatch(/1 protected/);

    const both = await compactThroughHook(raw, answers('drop_result', ['t1', 't2']));
    expect(both.final).not.toContain(typed);
    expect(count(both.final, TYPED)).toBe(1);
    expect(count(both.final, 'fast-jev-compaction truncated')).toBe(2);
    // Put back once, after the last of the two results.
    expect(indexOf(both.final, TYPED)).toBe(indexOf(both.final, '"tool_use_id":"b"'));
  });

  it('falls back to the built-in summary when the host gives no API view', async () => {
    const { raw } = session();
    const denied = await compactThroughHook(raw, answers('drop_call', ['t2']), () => ({ deny: 'no such agent' }));
    expect(denied.fellBack).toBe(true);
    expect(denied.logs.join('\n')).toMatch(/fallback.*no API view of the conversation \(no such agent\)/);

    const rowsOnly = await compactThroughHook(raw, answers('drop_call', ['t2']), () => host().rowsOf(raw));
    expect(rowsOnly.fellBack).toBe(true);
    expect(rowsOnly.logs.join('\n')).toMatch(/answers rows only/);
  });
});

describe('attachedContent', () => {
  const row = (id: string, text: string) => ({ role: 'user' as const, text: '', toolUses: [], toolResults: [{ tool_use_id: id, text, isError: false }] });
  const reminder = '<system-reminder>\n<total_tokens>5 tokens left</total_tokens>\n</system-reminder>';

  it('reads text folded into a result after its own (trimmed) text', () => {
    const view: ApiMessage[] = [
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'x', content: `53M\tdir\n\n${reminder}` }] },
    ];
    expect(attachedContent([row('x', ' 53M\tdir\n')], view)).toEqual([{ toolUseIds: ['x'], text: reminder }]);
  });

  it('reads blocks after the results, but not the text of a row of its own', () => {
    const prompt = { role: 'user' as const, text: 'go on', toolUses: [] };
    const view: ApiMessage[] = [
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'x', content: 'out' },
          { type: 'tool_result', tool_use_id: 'y', content: [{ type: 'text', text: 'out y' }] },
          { type: 'text', text: SKILL },
          { type: 'text', text: 'go on' },
        ],
      },
    ];
    expect(attachedContent([row('x', 'out'), row('y', 'out y'), prompt], view)).toEqual([
      { toolUseIds: ['x', 'y'], text: SKILL },
    ]);
    expect(attachedContent([row('x', 'out')], [{ role: 'user', content: [{ type: 'tool_result', tool_use_id: 'x', content: 'out' }] }])).toEqual([]);
  });

  it('marks content it cannot put back or attribute, and results it cannot see', () => {
    const image: ApiMessage = {
      role: 'user',
      content: [{ type: 'tool_result', tool_use_id: 'x', content: 'out' }, { type: 'image', source: {} }],
    };
    expect(attachedContent([row('x', 'out')], [image])).toEqual([{ toolUseIds: ['x'] }]);
    const changed: ApiMessage = { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'x', content: 'other' }] };
    expect(attachedContent([row('x', 'out')], [changed])).toEqual([{ toolUseIds: ['x'] }]);
    expect(attachedContent([row('x', 'out'), row('z', 'zz')], [image])).toEqual([{ toolUseIds: ['x'] }, { toolUseIds: ['z'] }]);
  });
});
