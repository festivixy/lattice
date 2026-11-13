// Runs suite.js inside a downloaded copy of VS Code, on a throwaway project folder.
const fs = require('fs');
const os = require('os');
const path = require('path');
const { runTests } = require('@vscode/test-electron');

const FILE = (name) => [
  `"""${name}: a small module for the typing test."""`,
  'import math',
  '',
  '',
  'def area(radius):',
  '    """Area of a circle."""',
  '    if radius < 0:',
  '        raise ValueError("negative radius")',
  '    return math.pi * radius ** 2',
  '',
  '',
  'def scale(values, factor):',
  '    return [value * factor for value in values]',
  '',
  '',
  'def describe(radius):',
  '    return f"r={radius:.2f} area={area(radius):.2f}"',
  '',
].join('\n');

async function main() {
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), 'lattice-typer-'));
  fs.writeFileSync(path.join(workspace, 'first.py'), FILE('first'));
  fs.writeFileSync(path.join(workspace, 'second.py'), FILE('second'));
  // the harness hands arguments to cmd.exe unquoted on Windows, which splits paths with spaces
  const arg = (p) => (process.platform === 'win32' ? `"${p}"` : p);
  const profile = path.resolve(__dirname, '..', '..', '.vscode-test');
  try {
    await runTests({
      version: 'insiders',  // a separate build, so it never collides with a VS Code you have open
      extensionDevelopmentPath: arg(path.resolve(__dirname, '..', '..')),
      extensionTestsPath: arg(path.resolve(__dirname, 'suite.js')),
      launchArgs: [
        arg(workspace), '--disable-extensions',
        `--user-data-dir=${arg(path.join(profile, 'user-data'))}`,
        `--extensions-dir=${arg(path.join(profile, 'extensions'))}`,
      ],
    });
  } finally {
    fs.rmSync(workspace, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
