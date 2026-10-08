import { describe, expect, it } from 'vitest';
import {
  attachedContent,
  classifyHidden,
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

  it('under drop_call, puts the typed message back as the user\'s words and keeps the skill call', async () => {
    const { raw, hidden } = session();
    const { out, final, toasts, logs } = await compactThroughHook(raw, answers('drop_call', ['t2', 't3']));
    expect(toasts).toEqual(['compaction done']);
    expect(out.messages).toBeDefined();
    expect(count(final, TYPED)).toBe(1);
    expect(count(final, SKILL)).toBe(1);
    expect(logs.join('\n')).toMatch(/1 protected, 1 attached put back/);
    // The prompt itself, as a user message, in place: after step 1, before step 3.
    const typedAt = indexOf(final, TYPED);
    expect(final[typedAt]).toMatchObject({ type: 'user', fresh: true, message: { content: TYPED } });
    expect(typedAt).toBeGreaterThan(indexOf(final, '"step 1"'));
    expect(typedAt).toBeLessThan(indexOf(final, SKILL));
    expect(count(final, '"step 2"')).toBe(0);
    // Step 2's token countdown went with it; the skill body stays where the host keeps it.
    expect(count(final, '899998 tokens left')).toBe(0);
    expect(final).toContain(hidden[1]);
    expect(count(final, '"skill":"deploy"')).toBe(1);
  });

  it('keeps the calls under drop_result, since their call rows (and what the host keeps there) survive', async () => {
    const { raw, hidden } = session();
    const { out, final, logs } = await compactThroughHook(raw, answers('drop_result', ['t2', 't3']));
    expect(out.messages).toBeDefined();
    expect(final).toContain(hidden[0]);
    expect(final).toContain(hidden[1]);
    expect(count(final, TYPED)).toBe(1);
    expect(count(final, SKILL)).toBe(1);
    expect(count(final, 'fast-jev-compaction truncated')).toBe(0);
    expect(logs.join('\n')).toMatch(/2 protected/);
  });

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
    const mixed = await compactThroughHook(raw, answers('drop_call', ['t1']));
    expect(mixed.final).toContain(typed);
    expect(count(mixed.final, '"step a"')).toBe(1);
    expect(mixed.logs.join('\n')).toMatch(/1 protected/);

    const both = await compactThroughHook(raw, answers('drop_call', ['t1', 't2']));
    expect(both.final).not.toContain(typed);
    expect(count(both.final, TYPED)).toBe(1);
    // Put back once, where the last of the two results was.
    expect(indexOf(both.final, TYPED)).toBe(indexOf(both.final, 'both done') - 1);
  });

  it('does not credit a following user row, or what the host keeps beside it, to the result', async () => {
    // A denied call: an error result, then the user's denial and feedback (one row,
    // two text blocks), then a reminder grouped under that row.
    const t = transcriptBuilder();
    const feedback = t.userBlocks('[Request interrupted by user for tool use]', 'use the staging bucket instead');
    const reminder = t.tokens(500);
    const raw: Raw[] = [
      t.prompt('Upload the build'),
      t.thinking(),
      t.use('up', 'mcp__s3__put', { bucket: 'prod' }),
      t.result('up', `The user doesn't want to proceed with this tool use. ${output(1)}`, true),
      feedback,
      reminder,
      t.thinking(),
      t.use('c', 'Bash', { command: 'step c' }),
      t.result('c', output(3)),
      t.say('done'),
    ];
    const { final, out } = await compactThroughHook(raw, answers('drop_call', ['t1']));
    expect(out.messages).toBeDefined();
    expect(count(final, 'use the staging bucket instead')).toBe(1);
    expect(count(final, '500 tokens left')).toBe(1);
    expect(final).toContain(feedback);
    expect(final).toContain(reminder);
    expect(count(final, '"bucket":"prod"')).toBe(0);
  });

  it('keeps content the host groups under a surviving call row', async () => {
    // A skill body between the call and its result belongs to the call row's group.
    const t = transcriptBuilder();
    const body = t.skillBody(SKILL);
    const raw: Raw[] = [
      t.prompt('Deploy'),
      t.thinking(),
      t.use('s', 'Skill', { skill: 'deploy' }),
      body,
      t.result('s', `Launching skill: deploy\n${output(1)}`),
      t.thinking(),
      t.use('c', 'Bash', { command: 'step c' }),
      t.result('c', output(3)),
      t.say('done'),
    ];
    const truncated = await compactThroughHook(raw, answers('drop_result', ['t1']));
    expect(truncated.final).toContain(body);
    expect(count(truncated.final, SKILL)).toBe(1);
    const dropped = await compactThroughHook(raw, answers('drop_call', ['t1']));
    expect(count(dropped.final, SKILL)).toBe(1);
  });

  it('keeps a ToolSearch call: its result loads deferred tools the built-in summary would carry over', async () => {
    const t = transcriptBuilder();
    const raw: Raw[] = [
      t.prompt('Read my mail'),
      t.thinking(),
      t.use('ts', 'ToolSearch', { query: 'gmail' }),
      t.toolSearch('ts', 'mcp__gmail__search_threads'),
      t.thinking(),
      t.use('g', 'mcp__gmail__search_threads', { q: 'invoice' }),
      t.result('g', output(1)),
      t.thinking(),
      t.use('c', 'Bash', { command: 'step c' }),
      t.result('c', output(3)),
      t.say('done'),
    ];
    const { final, logs } = await compactThroughHook(raw, answers('drop_call', ['t1', 't2']));
    expect(count(final, '"tool_name":"mcp__gmail__search_threads"')).toBe(1);
    expect(count(final, 'Tool loaded.')).toBe(0);
    expect(count(final, '"q":"invoice"')).toBe(0);
    expect(logs.join('\n')).toMatch(/1 call_dropped, 1 pinned, 1 protected/);
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

  const typed = (prompt: string) =>
    `<system-reminder>\nThe user sent a new message while you were working:\n${prompt}\n\nThis is how Claude Code surfaces messages the user sends mid-turn — within the running turn, often alongside the next tool result, rather than as a separate conversation turn. Address the message above as you continue this turn.\n</system-reminder>`;
  const folded = (...pieces: string[]): ApiMessage[] => [
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'x', content: ['53M\tdir', ...pieces].join('\n\n') }] },
  ];

  it('reads what is folded into a result after its own (trimmed) text: prompts back, countdowns go, the rest kept', () => {
    expect(attachedContent([row('x', ' 53M\tdir\n')], folded(reminder))).toEqual([]);
    expect(attachedContent([row('x', ' 53M\tdir\n')], folded(typed('homebrew is installed'), reminder))).toEqual([
      { toolUseIds: ['x'], text: 'homebrew is installed' },
    ]);
    expect(
      attachedContent([row('x', '53M\tdir')], folded(typed('a'), '<system-reminder>\n# Environment update\n</system-reminder>')),
    ).toEqual([{ toolUseIds: ['x'] }]);
    expect(attachedContent([row('x', '53M\tdir')], folded('Tool loaded.'))).toEqual([{ toolUseIds: ['x'] }]);
    expect(classifyHidden(`\n${typed('two\n\nparagraphs')}\n`)).toEqual({ prompt: 'two\n\nparagraphs' });
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
    // The skill body is not the user's words: those calls are kept.
    expect(attachedContent([row('x', 'out'), row('y', 'out y'), prompt], view)).toEqual([{ toolUseIds: ['x', 'y'] }]);
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

describe('images in tool results', () => {
  function screenshots() {
    const t = transcriptBuilder();
    const shot = (id: string): Raw => ({
      type: 'user',
      uuid: `shot-${id}`,
      toolUseResult: { type: 'image' },
      message: {
        content: [
          {
            type: 'tool_result',
            tool_use_id: id,
            content: [
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'iVBORw0KGgo=' } },
              { type: 'text', text: 'Screenshot taken' },
            ],
          },
        ],
      },
    });
    const raw: Raw[] = [t.prompt('Check the page')];
    for (const id of ['t1', 't2', 't3', 't4']) raw.push(t.thinking(), t.use(id, 'screenshot', {}), shot(id));
    raw.push(t.say('done'));
    return raw;
  }

  it('keeps a text-only tool\'s result that holds an image (a pasted picture folded in)', async () => {
    const t = transcriptBuilder();
    const raw: Raw[] = [t.prompt('Build it'), t.thinking(), t.use('b', 'Bash', { command: 'make' })];
    raw.push({
      type: 'user',
      uuid: 'folded',
      toolUseResult: { stdout: output(1) },
      message: {
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'b',
            content: [
              { type: 'text', text: output(1) },
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'iVBORw0KGgo=' } },
            ],
          },
        ],
      },
    });
    raw.push(t.thinking(), t.use('c', 'Bash', { command: 'step c' }), t.result('c', output(3)), t.say('done'));
    const { final, logs } = await compactThroughHook(raw, answers('drop_result', ['t1']));
    expect(final).toContain(raw.find((entry) => entry.uuid === 'folded'));
    expect(logs.join('\n')).toMatch(/1 protected/);
  });

  it('counts screenshots in the reduction and drops their images with the result', async () => {
    const raw = screenshots();
    const { final, out, logs } = await compactThroughHook(raw, answers('drop_result', ['t1', 't2']));
    expect(out.messages).toBeDefined();
    // Two of four screenshots go: about half the size, though their text is 16 chars.
    expect(logs.join('\n')).toMatch(/kept \d+\/\d+ messages, no summary \((4\d|5\d)% reduction; .*2 results truncated/);
    expect(count(final, 'removed 1 image from this tool result')).toBe(2);
    expect(count(final, '"media_type":"image/png"')).toBe(2);
    expect(final).toContain(raw.find((entry) => entry.uuid === 'shot-t3'));
  });
});
