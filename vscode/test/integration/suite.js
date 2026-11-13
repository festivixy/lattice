// Drives the extension the way a person would, inside a real VS Code (see run.js).
const assert = require('node:assert/strict');
const fs = require('fs');
const vscode = require('vscode');

const type = (text) => vscode.commands.executeCommand('type', { text });
const backspace = () => vscode.commands.executeCommand('latticeTyper.backspace', { fallback: 'deleteLeft' });
const active = () => vscode.window.activeTextEditor.document;
const draftTabs = () => vscode.window.tabGroups.all.flatMap((group) => group.tabs)
  .filter((tab) => tab.input instanceof vscode.TabInputText && tab.input.uri.scheme === 'draft');

async function run() {
  const folder = vscode.workspace.workspaceFolders[0].uri;
  const first = vscode.Uri.joinPath(folder, 'first.py');
  const original = fs.readFileSync(first.fsPath, 'utf8');
  await vscode.window.showTextDocument(first);

  await Promise.all([1, 2].map(() => vscode.commands.executeCommand('latticeTyper.toggle')));
  assert.equal(draftTabs().length, 1, 'a double press starts once');
  assert.equal(active().uri.scheme, 'draft', 'a copy is open');
  assert.match(active().uri.path, /\/first\.py$/);
  assert.equal(active().languageId, 'python');
  const opened = active().getText();
  assert.ok(opened.length > 0 && original.startsWith(opened), 'the copy opens partly written');
  const realTabs = vscode.window.tabGroups.all.flatMap((group) => group.tabs)
    .filter((tab) => tab.input instanceof vscode.TabInputText && tab.input.uri.scheme === 'file');
  assert.equal(realTabs.length, 0, 'the copy took the real tab\'s place');

  for (let i = 0; i < 20; i += 1) await type('q');
  const typed = active().getText();
  assert.ok(typed.length > opened.length && original.startsWith(typed), 'mashing types the real code');

  await backspace();
  assert.ok(active().getText().length < typed.length, 'backspace un-types');
  await type('k');
  assert.equal(active().getText(), typed, 'mashing again retypes the same code');

  await vscode.commands.executeCommand('latticeTyper.realTyping');
  await type('Z');
  assert.ok(active().getText().endsWith('Z'), 'F12 types what you press');
  await vscode.commands.executeCommand('latticeTyper.realTyping');
  await backspace();
  assert.equal(active().getText(), typed, 'an ordinary backspace still deletes');

  for (let i = 0; i < 1000 && /first\.py$/.test(active().uri.path); i += 1) await type('q');
  assert.equal(active().uri.scheme, 'draft');
  assert.match(active().uri.path, /\/second\.py$/, 'a finished file moves on to the next');
  assert.equal(draftTabs().length, 1, 'the finished copy closed without a prompt');

  const secondCopy = active().uri.toString();
  for (let i = 0; i < 1000 && active().uri.toString() === secondCopy; i += 1) await type('q');
  assert.notEqual(active().uri.toString(), secondCopy, 'the last file finishing starts the next round');
  assert.equal(draftTabs().length, 1);

  await vscode.commands.executeCommand('latticeTyper.toggle');
  assert.equal(draftTabs().length, 0, 'stopping closes the copies');
  assert.equal(active().uri.toString(), first.toString(), 'the real file is back in its place');
  assert.equal(fs.readFileSync(first.fsPath, 'utf8'), original, 'the real file was never changed');

  const scratch = await vscode.workspace.openTextDocument({ content: '' });
  await vscode.window.showTextDocument(scratch);
  await type('hi');
  assert.equal(scratch.getText(), 'hi', 'typing is back to normal');
}

module.exports = { run };
