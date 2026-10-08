// A copy of what Claude Code 2.1.292 does around a `session.compact` hook,
// for tests: how it turns the transcript into the rows the hook gets
// (`rowsOf`), how it turns the rows the hook hands back into the transcript
// (`messagesOf`), and what `$.session.messages({ as: 'api' })` answers.
//
// `rowsOf`/`messagesOf` are transcribed from the minified 2.1.292 binary
// (Une, $ne, zpt, Hne, Wpt, gB, Doo/Moo); the names in the comments are the
// minified ones. The Agent-tool extras of Hne (agentId, durationMs) are left
// out. `apiViewOf` is a model of the host's normalization, written from what
// 2.1.292 answered for a real transcript: consecutive user-side messages
// merge into one, tool results first; trailing <system-reminder> texts fold
// into the last tool result's text after a blank line; other texts (a skill
// body) stay blocks of their own.

import type { ApiBlock, ApiMessage } from '../hooks/fast-jev.ts';

export type RawBlock = { type: string; [field: string]: unknown };
export type Raw = {
  type: 'user' | 'assistant' | 'attachment' | 'system';
  uuid: string;
  isMeta?: boolean;
  isVirtual?: boolean;
  toolUseResult?: unknown;
  message?: { content: string | RawBlock[] };
  attachment?: { type: string; [field: string]: unknown };
};

export type Row = {
  role: 'user' | 'assistant';
  text: string;
  toolUses: { tool_use_id: string; tool: string; input: Record<string, unknown>; text?: string; isError?: true; result?: unknown }[];
  toolResults?: { tool_use_id: string; text: string; isError: boolean; result?: unknown }[];
  handle?: string;
};

/** A message the host builds from a handle-less row (Wpt). */
export type Fresh = { type: 'user' | 'assistant'; fresh: true; message: { content: string | RawBlock[] } };

// Moo / Doo: a tool_result's content as text, text blocks only.
const textOf = (content: unknown): string =>
  typeof content === 'string'
    ? content
    : Array.isArray(content)
      ? content
          .flatMap((block) =>
            typeof block === 'object' && block !== null && block.type === 'text' ? [String(block.text ?? '')] : [],
          )
          .join('\n')
      : '';

const userRow = (text: string, results: NonNullable<Row['toolResults']>): Row => ({
  role: 'user',
  text,
  toolUses: [],
  ...(results.length > 0 && { toolResults: results }),
});

// $ne: undefined for every attachment and for meta or virtual user messages.
export function rowOf(entry: Raw): Row | undefined {
  if (entry.type === 'assistant') {
    const uses: Row['toolUses'] = [];
    const texts: string[] = [];
    for (const block of entry.message!.content as RawBlock[]) {
      if (block.type === 'text') texts.push(String(block.text));
      else if (block.type === 'tool_use') {
        uses.push({ tool_use_id: String(block.id), tool: String(block.name), input: structuredClone(block.input as Record<string, unknown>) });
      }
    }
    return { role: 'assistant', text: texts.join(''), toolUses: uses };
  }
  if (entry.type !== 'user') return undefined;
  if (entry.isMeta === true || entry.isVirtual === true) return undefined;
  const content = entry.message!.content;
  if (typeof content === 'string') return userRow(content, []);
  const texts: string[] = [];
  const results: NonNullable<Row['toolResults']> = [];
  const stored = entry.toolUseResult !== undefined;
  for (const block of content) {
    if (block.type === 'text') texts.push(String(block.text));
    else if (block.type === 'tool_result') {
      results.push({
        tool_use_id: String(block.tool_use_id),
        text: textOf(block.content),
        isError: block.is_error === true,
        ...(stored && { result: entry.toolUseResult }),
      });
    }
  }
  return userRow(texts.join(''), results);
}

// Hne + zpt: a row's tool uses carry their outcome, and the row its handle.
const withOutcome = (row: Row, results: Map<string, NonNullable<Row['toolResults']>[number]>, handle: string): Row => ({
  ...row,
  toolUses: row.toolUses.map((use) => {
    const found = results.get(use.tool_use_id);
    return {
      ...use,
      ...(found !== undefined && 'result' in found && { result: found.result }),
      ...(found !== undefined && { text: found.text }),
      ...(found?.isError === true && { isError: true as const }),
    };
  }),
  handle,
});

// gB: what decides whether a handed-back row is still the engine's own.
const sameness = (row: Row): string =>
  JSON.stringify({ role: row.role, text: row.text, toolUses: row.toolUses, toolResults: row.toolResults });

// Wpt: a row without a (matching) handle becomes one fresh message.
export function freshOf(row: Row): Fresh {
  const text = row.text === '' ? [] : [{ type: 'text', text: row.text }];
  if (row.role === 'assistant') {
    const content = [
      ...text,
      ...row.toolUses.map((use) => ({ type: 'tool_use', id: use.tool_use_id, name: use.tool, input: use.input })),
    ];
    return { type: 'assistant', fresh: true, message: { content: content.length === 0 ? '' : content } };
  }
  const results = (row.toolResults ?? []).map((result) => ({
    type: 'tool_result',
    tool_use_id: result.tool_use_id,
    content: result.text,
    ...(result.isError && { is_error: true }),
  }));
  return { type: 'user', fresh: true, message: { content: results.length === 0 ? row.text : [...results, ...text] } };
}

// Une: one compaction's rowsOf/messagesOf pair.
export function host() {
  const groups = new Map<string, { row: Row; messages: Raw[] }>();
  return {
    rowsOf(entries: readonly Raw[]): Row[] {
      const results = new Map<string, NonNullable<Row['toolResults']>[number]>();
      for (const entry of entries) for (const result of rowOf(entry)?.toolResults ?? []) results.set(result.tool_use_id, result);
      const rows: Row[] = [];
      let before: Raw[] = [];
      let group: Raw[] = [];
      let started = false;
      for (const entry of entries) {
        const row = rowOf(entry);
        // A message the rows cannot show joins the group of the row before it.
        if (row === undefined && started) {
          group.push(entry);
          continue;
        }
        if (row === undefined) {
          before.push(entry);
          continue;
        }
        const handled = withOutcome(row, results, entry.uuid);
        group = [...before, entry];
        before = [];
        started = true;
        groups.set(entry.uuid, { row: handled, messages: group });
        rows.push(handled);
      }
      return rows;
    },
    messagesOf(rows: readonly Row[]): (Raw | Fresh)[] {
      const used = new Set<string>();
      return rows.flatMap((row) => {
        const own = row.handle === undefined ? undefined : groups.get(row.handle);
        if (!(own !== undefined && !used.has(own.row.handle ?? '') && sameness(own.row) === sameness(row))) {
          return [freshOf(row)];
        }
        used.add(own.row.handle ?? '');
        return own.messages;
      });
    },
  };
}

/** Model of the API form 2.1.292 answers for `$.session.messages({ as: 'api' })` (see the header). */
export function apiViewOf(entries: readonly Raw[]): ApiMessage[] {
  const view: ApiMessage[] = [];
  const add = (role: ApiMessage['role'], blocks: ApiBlock[]): void => {
    if (blocks.length === 0) return;
    const last = view[view.length - 1];
    if (last && last.role === role) {
      // ZHr: merging text after text adds a newline to the earlier block.
      const tail = last.content[last.content.length - 1];
      if (role === 'user' && tail?.type === 'text' && blocks[0]?.type === 'text') {
        last.content[last.content.length - 1] = { ...tail, text: `${String(tail.text)}\n` };
      }
      last.content.push(...blocks.map((block) => ({ ...block })));
    } else view.push({ role, content: blocks.map((block) => ({ ...block })) });
  };
  for (const entry of entries) {
    if (entry.type === 'assistant') {
      add('assistant', (entry.message!.content as RawBlock[]).filter((b) => b.type === 'text' || b.type === 'tool_use'));
    } else if (entry.type === 'user') {
      const content = entry.message!.content;
      add('user', typeof content === 'string' ? [{ type: 'text', text: content }] : content);
    } else if (entry.type === 'attachment') {
      add('user', renderAttachment(entry.attachment!));
    }
  }
  for (const message of view) {
    if (message.role !== 'user') continue;
    const results = message.content.filter((b) => b.type === 'tool_result');
    if (results.length === 0) continue;
    const rest = message.content.filter((b) => b.type !== 'tool_result');
    const foldable =
      rest.length > 0 && rest.every((b) => b.type === 'text' && String(b.text).startsWith('<system-reminder>'));
    // iXe: the result's text trimmed, then each reminder trimmed, after blank lines.
    if (foldable) {
      const last = results[results.length - 1]!;
      const own = typeof last.content === 'string' ? last.content : textOf(last.content);
      results[results.length - 1] = {
        ...last,
        content: [own.trim(), ...rest.map((b) => String(b.text).trim())].filter(Boolean).join('\n\n'),
      };
      message.content = results;
    } else {
      message.content = [...results, ...rest];
    }
  }
  return view;
}

function renderAttachment(attachment: { type: string; [field: string]: unknown }): ApiBlock[] {
  switch (attachment.type) {
    case 'queued_command':
      return [
        {
          type: 'text',
          text: `<system-reminder>\nThe user sent a new message while you were working:\n${String(attachment.prompt)}\n\nThis is how Claude Code surfaces messages the user sends mid-turn.\n</system-reminder>`,
        },
      ];
    case 'total_tokens_reminder':
      return [{ type: 'text', text: `<system-reminder>\n${String(attachment.text)}\n</system-reminder>` }];
    case 'pasted_image':
      return [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'iVBORw0KGgo=' } }];
    default:
      return [];
  }
}

/** Builds raw transcript entries with unique uuids. */
export function transcriptBuilder() {
  let n = 0;
  const uuid = () => `u${(n += 1)}`;
  return {
    prompt: (text: string): Raw => ({ type: 'user', uuid: uuid(), message: { content: text } }),
    thinking: (): Raw => ({ type: 'assistant', uuid: uuid(), message: { content: [{ type: 'thinking', thinking: '…' }] } }),
    say: (text: string): Raw => ({ type: 'assistant', uuid: uuid(), message: { content: [{ type: 'text', text }] } }),
    use: (id: string, tool: string, input: Record<string, unknown>): Raw => ({
      type: 'assistant',
      uuid: uuid(),
      message: { content: [{ type: 'tool_use', id, name: tool, input }] },
    }),
    result: (id: string, text: string, isError = false): Raw => ({
      type: 'user',
      uuid: uuid(),
      toolUseResult: isError ? text : { stdout: text },
      message: { content: [{ type: 'tool_result', tool_use_id: id, content: text, ...(isError && { is_error: true }) }] },
    }),
    /** A user message of several text blocks (a denial with the user's feedback). */
    userBlocks: (...texts: string[]): Raw => ({
      type: 'user',
      uuid: uuid(),
      message: { content: texts.map((text) => ({ type: 'text', text })) },
    }),
    queued: (prompt: string): Raw => ({ type: 'attachment', uuid: uuid(), attachment: { type: 'queued_command', prompt } }),
    tokens: (left: number): Raw => ({
      type: 'attachment',
      uuid: uuid(),
      attachment: { type: 'total_tokens_reminder', text: `<total_tokens>${left} tokens left</total_tokens>` },
    }),
    image: (): Raw => ({ type: 'attachment', uuid: uuid(), attachment: { type: 'pasted_image' } }),
    skillBody: (text: string): Raw => ({ type: 'user', uuid: uuid(), isMeta: true, message: { content: text } }),
  };
}
