const assert = require('node:assert/strict');
const { test } = require('node:test');
const { Draft, commentFor } = require('../draft');

const SOURCE = 'def f(x):\n    y = x + 1\n    return y\n\n\ndef g():\n    pass\n';

// A plain string standing in for the editor: applies each edit and moves the cursor after it.
class Doc {
  constructor(draft) {
    this.draft = draft;
    this.text = draft.text;
    this.cursor = draft.anchor;
  }

  apply(edit) {
    if (!edit) return false;
    this.change(edit.at, edit.remove, edit.insert);
    this.cursor = edit.at + edit.insert.length;
    return true;
  }

  change(at, remove, insert) {  // what VS Code reports for any edit, ours or the user's
    this.text = this.text.slice(0, at) + insert + this.text.slice(at + remove);
    this.draft.applyChange(at, remove, insert.length);
  }

  mash(times = 1) {
    for (let i = 0; i < times; i += 1) this.apply(this.draft.mash(this.text, this.cursor));
  }

  backspace() {
    return this.apply(this.draft.backspace(this.cursor));
  }

  lines() {
    return this.text.split('\n');
  }
}

test('opens partly written with the cursor where the writing stops', () => {
  const draft = new Draft(SOURCE);
  assert.equal(draft.text, 'def f(x):\n    y = x + 1\n    return y');
  assert.equal(draft.anchor, draft.text.length);
});

test('mashing types the rest of the file exactly, then reports it is done', () => {
  const doc = new Doc(new Draft(SOURCE));
  while (!doc.draft.done) doc.mash();
  assert.equal(doc.text, SOURCE);
  assert.equal(doc.draft.mash(doc.text, doc.cursor), null);
});

test('a new line comes with its indentation in one keypress', () => {
  const doc = new Doc(new Draft('a:\n    b\n    c\n'));  // opens with "a:" written
  doc.mash();
  assert.equal(doc.text, 'a:\n    ');
});

test('windows line endings type as plain newlines', () => {
  const doc = new Doc(new Draft('a\r\nb\r\nc\r\nd\r\n'));
  while (!doc.draft.done) doc.mash();
  assert.equal(doc.text, 'a\nb\nc\nd\n');
});

test('backspace un-types and mashing again retypes the same code', () => {
  const doc = new Doc(new Draft(SOURCE));
  doc.mash(4);
  const typed = doc.text;
  assert.ok(doc.backspace());
  assert.equal(doc.text.length, typed.length - 1);
  doc.mash();
  assert.equal(doc.text, typed);
});

test('backspace after the cursor moved is left to the editor', () => {
  const doc = new Doc(new Draft(SOURCE));
  doc.mash(2);
  doc.cursor -= 1;
  assert.equal(doc.draft.backspace(doc.cursor), null);
});

test('mashing away from the write point types a comment on its own line', () => {
  const doc = new Doc(new Draft(SOURCE));
  doc.draft.comment = commentFor('python');
  doc.cursor = doc.text.indexOf('y = x');  // middle of line 2
  doc.mash(12);
  assert.deepEqual(doc.lines().slice(1, 4), ['    y = x + 1', '    # TODO: han', '    return y']);
});

test('one step left of the write point still types a whole comment, then code resumes', () => {
  const doc = new Doc(new Draft(SOURCE));
  doc.draft.comment = commentFor('python');
  doc.cursor -= 1;
  doc.mash(80);  // the whole comment and then some real code
  const lines = doc.lines();
  assert.match(lines[3], /^    # TODO: .+/);
  assert.equal(lines.filter((line) => line.includes('#')).length, 1);
  assert.equal(doc.text.replace(/\n    # .*/, ''), SOURCE);
});

test('returning to the write point picks the real code back up', () => {
  const doc = new Doc(new Draft(SOURCE));
  doc.cursor = 3;
  doc.mash(5);  // a comment under the first line
  doc.cursor = doc.draft.anchor;
  while (!doc.draft.done) doc.mash();
  assert.equal(doc.lines().filter((line) => !line.includes('//')).join('\n'), SOURCE);
});

test('the user typing at the write point keeps the code going after it', () => {
  const doc = new Doc(new Draft(SOURCE));
  doc.change(doc.cursor, 0, 'Z');
  doc.cursor += 1;
  doc.mash(2);
  assert.ok(doc.text.endsWith('return yZ\n\n'));
});

test('deleting before the write point shifts it back', () => {
  const doc = new Doc(new Draft(SOURCE));
  doc.change(0, 4, '');  // "def "
  doc.cursor = doc.draft.anchor;
  doc.mash();
  assert.ok(doc.text.endsWith('return y\n'));
  assert.equal(doc.draft.anchor, doc.text.length);
});

test('backspacing back through code into a comment retypes both in order', () => {
  const doc = new Doc(new Draft(SOURCE));
  doc.cursor -= 1;
  doc.mash(52);  // a whole comment, then a few characters of code
  const typed = doc.text;
  for (let i = 0; i < 10; i += 1) assert.ok(doc.backspace());
  doc.mash(10);
  assert.equal(doc.text, typed);
});

test('an emoji types in one keypress, never half of one', () => {
  const doc = new Doc(new Draft('a\nb\n"🚀 go"\n'));
  doc.mash(5);  // opens with "a": then "\n", "b", "\n", '"', the emoji
  assert.ok(doc.text.endsWith('"🚀'));
  assert.ok(doc.backspace());
  assert.ok(doc.text.endsWith('"'));
});

test('comment syntax follows the language', () => {
  assert.deepEqual(commentFor('python'), { start: '#', end: '' });
  assert.deepEqual(commentFor('typescript'), { start: '//', end: '' });
  assert.deepEqual(commentFor('sql'), { start: '--', end: '' });
  assert.deepEqual(commentFor('html'), { start: '<!--', end: '-->' });
  assert.deepEqual(commentFor('something-new'), { start: '//', end: '' });
});
