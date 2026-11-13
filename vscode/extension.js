/* Fake typing in VS Code.

Ctrl+Alt+Y (Ctrl+Cmd+Y on a Mac) swaps the open project file for a copy of it that is partly
written. From then on every letter you press there types the next character of the real file,
so mashing comes out as clean code; mashing away from where the writing left off types
believable comments instead (see draft.js). When a file is finished the next one from the
project opens. F12 switches to typing exactly what you press. Backspace and Ctrl+Z right after
mashing un-type, so mashing again retypes the same code.

The real files are never touched: copies live only in memory, under the draft: scheme, and
disappear when the shortcut is pressed again.
*/
const path = require('path');
const vscode = require('vscode');
const { Draft, commentFor } = require('./draft');

const SCHEME = 'draft';
const MAX_FILES = 300;
const MAX_BYTES = 200000;  // bigger files are skipped when picking the next one
const MIN_LINES = 8;
const MAX_LINE_LENGTH = 120;  // on average; more than this is minified or generated
const EXCLUDE = '**/{node_modules,.git,dist,build,out,.venv,venv,__pycache__,target,.vscode-test}/**';
const CODE_FILES = '**/*.{py,js,ts,tsx,jsx,go,rs,java,kt,c,cc,cpp,h,hpp,cs,rb,php,swift,lua,sh}';

/** In-memory files behind the copies, so nothing is ever written to disk. */
class MemoryFs {
  constructor() {
    this.files = new Map();
    this.emitter = new vscode.EventEmitter();
    this.onDidChangeFile = this.emitter.event;
  }

  watch() {
    return new vscode.Disposable(() => {});
  }

  stat(uri) {
    const file = this.files.get(uri.toString());
    if (file) return { type: vscode.FileType.File, ctime: file.ctime, mtime: file.mtime, size: file.data.byteLength };
    const folder = uri.path.replace(/\/?$/, '/');
    if ([...this.files.values()].some((other) => other.uri.path.startsWith(folder))) {
      return { type: vscode.FileType.Directory, ctime: 0, mtime: 0, size: 0 };
    }
    throw vscode.FileSystemError.FileNotFound(uri);
  }

  readDirectory() {
    return [];
  }

  createDirectory() {}

  readFile(uri) {
    const file = this.files.get(uri.toString());
    if (!file) throw vscode.FileSystemError.FileNotFound(uri);
    return file.data;
  }

  writeFile(uri, data) {
    const old = this.files.get(uri.toString());
    const now = Date.now();
    this.files.set(uri.toString(), { uri, data, ctime: old ? old.ctime : now, mtime: now });
    this.emitter.fire([{ type: old ? vscode.FileChangeType.Changed : vscode.FileChangeType.Created, uri }]);
  }

  delete(uri) {
    this.files.delete(uri.toString());
    this.emitter.fire([{ type: vscode.FileChangeType.Deleted, uri }]);
  }

  rename(from, to) {
    this.writeFile(to, this.readFile(from));
    this.delete(from);
  }
}

const memory = new MemoryFs();
let session = null;
let busy = false;  // a second press while starting or stopping is ignored

function activate(context) {
  context.subscriptions.push(
    vscode.workspace.registerFileSystemProvider(SCHEME, memory, { isCaseSensitive: true }),
    vscode.commands.registerCommand('latticeTyper.toggle', toggle),
    vscode.commands.registerCommand('latticeTyper.realTyping', toggleReal),
    vscode.commands.registerCommand('latticeTyper.backspace', (args) => backspace((args && args.fallback) || 'deleteLeft')),
    { dispose: () => session && session.disposables.forEach((d) => d.dispose()) },
  );
}

function deactivate() {}

// --- starting and stopping ------------------------------------------------------------------

async function toggle() {
  if (busy) return;
  busy = true;
  try {
    await (session ? stop() : start());
  } finally {
    busy = false;
  }
}

async function start() {
  const editor = vscode.window.activeTextEditor;
  const origin = editor && editor.document.uri.scheme === 'file' ? editor.document.uri : undefined;
  const files = await projectFiles(origin);
  if (!origin && !files.length) {
    vscode.window.showInformationMessage('Open a file from your project first.');
    return;
  }
  let typing;
  try {
    typing = vscode.commands.registerCommand('type', onType);
  } catch {
    vscode.window.showErrorMessage('Another extension (Vim?) already handles typing, so this cannot start.');
    return;
  }
  const s = {
    drafts: new Map(), real: false, queue: Promise.resolve(), next: 0, count: 0, origin: undefined,
    files: files.length ? files : [origin],  // a one-file project starts that file over
    disposables: [typing, vscode.workspace.onDidChangeTextDocument(onChange)],
  };
  session = s;
  await setContext(true, false);
  const column = editor ? editor.viewColumn : vscode.ViewColumn.Active;
  if (origin && (await openDraft(s, origin, column))) {
    const tabs = tabsFor(origin).filter((tab) => !tab.isDirty);  // the copy takes the real file's place
    if (tabs.length) {
      s.origin = { uri: origin, column };
      await vscode.window.tabGroups.close(tabs);
    }
  } else {
    await nextFile(s, column);
  }
}

async function stop() {
  const s = session;
  session = null;
  s.disposables.forEach((d) => d.dispose());
  await setContext(false, false);
  await s.queue;
  for (const doc of vscode.workspace.textDocuments.filter((d) => d.uri.scheme === SCHEME)) {
    await discard(doc);
  }
  if (s.origin) await vscode.window.showTextDocument(s.origin.uri, { viewColumn: s.origin.column, preview: false });
}

function setContext(active, real) {
  return Promise.all([
    vscode.commands.executeCommand('setContext', 'latticeTyper.active', active),
    vscode.commands.executeCommand('setContext', 'latticeTyper.real', real),
  ]);
}

// --- files --------------------------------------------------------------------------------

/** Other files to move on to: the same kind as the one open first, else any code. */
async function projectFiles(origin) {
  const ext = origin ? path.posix.extname(origin.path) : '';
  let uris = ext ? await vscode.workspace.findFiles(`**/*${ext}`, EXCLUDE, MAX_FILES) : [];
  if (uris.length < 2) uris = await vscode.workspace.findFiles(CODE_FILES, EXCLUDE, MAX_FILES);
  const others = uris.filter((uri) => !origin || uri.toString() !== origin.toString());
  for (let i = others.length - 1; i > 0; i -= 1) {  // shuffle
    const j = Math.floor(Math.random() * (i + 1));
    [others[i], others[j]] = [others[j], others[i]];
  }
  return others;
}

/** Opens an in-memory copy of `real`, partly written. With `picky`, skips files that would look odd. */
async function openDraft(s, real, column, picky = false) {
  let source;
  try {
    source = new TextDecoder().decode(await vscode.workspace.fs.readFile(real));
  } catch {
    return false;
  }
  const lines = source.split('\n').length;
  if (picky && (source.includes('\0') || source.length > MAX_BYTES || lines < MIN_LINES
      || source.length / lines > MAX_LINE_LENGTH)) {
    return false;
  }
  const draft = new Draft(source);
  s.count += 1;
  const name = vscode.workspace.asRelativePath(real, false).replace(/\\/g, '/');
  const uri = vscode.Uri.from({ scheme: SCHEME, path: `/${name}`, query: String(s.count) });
  memory.writeFile(uri, new TextEncoder().encode(draft.text));
  const doc = await vscode.workspace.openTextDocument(uri);
  draft.comment = commentFor(doc.languageId);
  s.drafts.set(uri.toString(), draft);
  const editor = await vscode.window.showTextDocument(doc, { viewColumn: column, preview: false });
  const end = doc.positionAt(draft.anchor);
  editor.selection = new vscode.Selection(end, end);
  editor.revealRange(editor.selection, vscode.TextEditorRevealType.InCenterIfOutsideViewport);
  return true;
}

async function nextFile(s, column, previous) {
  for (let tries = 0; tries <= s.files.length; tries += 1) {
    const real = s.files[s.next % s.files.length];
    s.next += 1;
    const picky = tries < s.files.length;  // if nothing looks right, the last try takes the next file as is
    if (await openDraft(s, real, column, picky)) {
      if (previous) await discard(previous, s);
      return;
    }
  }
}

/** Closes a copy without a "save changes?" prompt: saving only writes to memory. */
async function discard(doc, s) {
  if (doc.isDirty) await doc.save();
  const tabs = tabsFor(doc.uri);
  if (tabs.length) await vscode.window.tabGroups.close(tabs);
  if (s) s.drafts.delete(doc.uri.toString());
  try {
    memory.delete(doc.uri);
  } catch {
    // already gone
  }
}

function tabsFor(uri) {
  return vscode.window.tabGroups.all.flatMap((group) => group.tabs)
    .filter((tab) => tab.input instanceof vscode.TabInputText && tab.input.uri.toString() === uri.toString());
}

// --- keys ---------------------------------------------------------------------------------

/** Every typed character in VS Code comes here while running; only copies are touched. */
function onType(args) {
  const s = session;
  const editor = vscode.window.activeTextEditor;
  if (!s || !editor || !s.drafts.has(editor.document.uri.toString())) {
    return vscode.commands.executeCommand('default:type', args);
  }
  if (s.real || args.text === '\n') return enqueue(s, () => vscode.commands.executeCommand('default:type', args));
  return enqueue(s, () => mash(s, editor));
}

/** Keys are handled one at a time, in order, however fast they come in. */
function enqueue(s, job) {
  s.queue = s.queue.then(job).catch((error) => console.error('lattice-typer:', error));
  return s.queue;
}

async function mash(s, editor) {
  const doc = editor.document;
  const draft = s.drafts.get(doc.uri.toString());
  if (session !== s || !draft || doc.isClosed) return;
  const edit = draft.mash(doc.getText(), doc.offsetAt(editor.selection.active));
  if (!edit) await nextFile(s, editor.viewColumn, doc);
  else if (!(await apply(editor, edit))) draft.backspace(draft.lastEnd);  // it never landed: un-type it
}

function backspace(fallback) {
  const s = session;
  const editor = vscode.window.activeTextEditor;
  const draft = s && editor && s.drafts.get(editor.document.uri.toString());
  if (!draft) return vscode.commands.executeCommand(fallback);
  return enqueue(s, async () => {
    const doc = editor.document;
    if (session !== s || doc.isClosed || !s.drafts.has(doc.uri.toString())) return;  // file moved on meanwhile
    const edit = editor.selection.isEmpty ? draft.backspace(doc.offsetAt(editor.selection.active)) : null;
    if (edit) await apply(editor, edit);
    else await vscode.commands.executeCommand(fallback);
  });
}

async function apply(editor, { at, remove, insert }) {
  const doc = editor.document;
  const range = new vscode.Range(doc.positionAt(at), doc.positionAt(at + remove));
  if (!(await editor.edit((builder) => builder.replace(range, insert)))) return false;
  const end = doc.positionAt(at + insert.length);
  editor.selection = new vscode.Selection(end, end);
  editor.revealRange(editor.selection);
  return true;
}

function toggleReal() {
  if (!session) return undefined;
  session.real = !session.real;
  return setContext(true, session.real);
}

function onChange(event) {
  const draft = session && session.drafts.get(event.document.uri.toString());
  if (!draft) return;
  [...event.contentChanges]
    .sort((a, b) => b.rangeOffset - a.rangeOffset)  // later edits first, so earlier offsets stay valid
    .forEach((change) => draft.applyChange(change.rangeOffset, change.rangeLength, change.text.length));
}

module.exports = { activate, deactivate };
