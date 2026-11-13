/* One file being "written": the same idea as lattice's editor pane (core/editor.py).

The copy opens partly written and remembers where its writing left off (the anchor). Each
mash there types the next character of the real file, a newline bringing the next line's
indentation with it, so mashing comes out as clean code. Mashing anywhere else starts a new
line and types a believable comment instead. Nothing here knows about VS Code: it works on
plain text and offsets, and returns edits for the caller to apply.
*/
const PREFILL_SHARE = 0.45;  // of a file's lines already written when it opens
const NOTES = [
  'TODO: handle the empty case before this ships',
  'NOTE: order matters here, the caller relies on it',
  'FIXME: revisit after the next profiling pass',
  'perf: cache this, it shows up hot in the profiler',
  'keep this in sync with the config defaults',
  'TODO: add a regression test for this path',
  'NOTE: intentionally not async, see the call site',
  'edge case: this can run before init finishes',
];
const LINE_COMMENTS = {
  '#': ['python', 'shellscript', 'ruby', 'perl', 'r', 'powershell', 'yaml', 'toml', 'makefile',
    'dockerfile', 'elixir', 'julia', 'coffeescript', 'nim', 'cmake', 'graphql'],
  '--': ['sql', 'lua', 'haskell', 'elm', 'ada'],
  '%': ['latex', 'tex', 'matlab', 'erlang', 'bibtex'],
  ';': ['clojure', 'lisp', 'scheme', 'ini'],
};
const BLOCK_COMMENTS = { html: ['<!--', '-->'], xml: ['<!--', '-->'], markdown: ['<!--', '-->'], css: ['/*', '*/'] };

/** @returns {{start: string, end: string}} how a comment looks in a VS Code language */
function commentFor(languageId) {
  if (BLOCK_COMMENTS[languageId]) {
    const [start, end] = BLOCK_COMMENTS[languageId];
    return { start, end };
  }
  const start = Object.keys(LINE_COMMENTS).find((mark) => LINE_COMMENTS[mark].includes(languageId));
  return { start: start || '//', end: '' };
}

/** @typedef {{at: number, remove: number, insert: string}} Edit */

class Draft {
  constructor(source, comment = commentFor('')) {
    this.source = source.replace(/\r\n?/g, '\n');
    this.comment = comment;
    const lines = this.source.split('\n');
    const written = Math.max(1, Math.floor(lines.length * PREFILL_SHARE));
    this.text = lines.slice(0, written).join('\n');  // what the copy opens with
    this.pos = this.text.length;  // next character of the source to type
    this.anchor = this.pos;  // where the writing left off, as an offset into the document
    this.typed = [];  // [{chunk, fromSource}] typed since the cursor last moved
    this.note = '';  // rest of the comment being typed away from the anchor
    this.notesUsed = 0;
    this.lastEnd = -1;  // cursor offset right after our last edit
  }

  get done() {
    return this.pos >= this.source.length;
  }

  /**
   * One keypress at `cursor`. Returns null when the file is finished and the cursor is at the
   * write point (time for the next file).
   * @returns {Edit | null}
   */
  mash(text, cursor) {
    if (cursor !== this.lastEnd) this.forget();
    if (!this.note && cursor === this.anchor) {
      if (this.done) return null;
      let size = this.source.codePointAt(this.pos) > 0xffff ? 2 : 1;  // emoji are two code units
      if (this.source[this.pos] === '\n') {
        while (/[ \t]/.test(this.source[this.pos + size] || '')) size += 1;
      }
      const chunk = this.source.slice(this.pos, this.pos + size);
      this.pos += size;
      return this.record(cursor, chunk, true);
    }
    let at = cursor;
    if (!this.note) {  // notes go on a line of their own, under the cursor's line
      const lineStart = cursor > 0 ? text.lastIndexOf('\n', cursor - 1) + 1 : 0;
      const newline = text.indexOf('\n', cursor);
      at = newline === -1 ? text.length : newline;
      const indent = /^[ \t]*/.exec(text.slice(lineStart, at))[0];
      const { start, end } = this.comment;
      const words = NOTES[this.notesUsed % NOTES.length];
      this.notesUsed += 1;
      this.note = `\n${indent}${start} ${words}${end ? ` ${end}` : ''}`;
    }
    const size = this.note[0] === '\n' ? 1 + /^[ \t]*/.exec(this.note.slice(1))[0].length : 1;
    const chunk = this.note.slice(0, size);
    this.note = this.note.slice(size);
    return this.record(at, chunk, false);
  }

  /**
   * Right after mashing, backspace un-types the last chunk so mashing again retypes it.
   * Returns null when it should be an ordinary delete.
   * @returns {Edit | null}
   */
  backspace(cursor) {
    if (cursor !== this.lastEnd || !this.typed.length) {
      this.forget();
      return null;
    }
    const { chunk, fromSource } = this.typed.pop();
    if (fromSource) this.pos -= chunk.length;
    else this.note = chunk + this.note;
    this.lastEnd = cursor - chunk.length;
    return { at: this.lastEnd, remove: chunk.length, insert: '' };
  }

  /** Keeps the anchor in place across any edit to the document, ours or the user's. */
  applyChange(offset, removed, inserted) {
    if (offset + removed <= this.anchor) this.anchor += inserted - removed;
    else if (offset < this.anchor) this.anchor = offset + inserted;
  }

  forget() {
    this.typed = [];
    this.note = '';
  }

  record(at, chunk, fromSource) {
    this.typed.push({ chunk, fromSource });
    this.lastEnd = at + chunk.length;
    return { at, remove: 0, insert: chunk };
  }
}

module.exports = { Draft, commentFor };
