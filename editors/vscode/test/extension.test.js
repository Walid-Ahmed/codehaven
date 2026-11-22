// The extension's commands, run against a small stand-in for the VS Code API. Run with: npm test
"use strict";
const test = require("node:test");
const assert = require("node:assert");
const Module = require("module");

// Load extension.js with a fake "vscode" module that records what the extension asks VS Code to do.
function load(settings = {}) {
  const sent = [], registered = {};
  const fake = {
    sent, registered, provider: null,
    window: {
      terminals: [], activeTextEditor: null,
      createTerminal: (options) => ({name: options.name, show() {}, sendText: (text) => sent.push(text)}),
      showWarningMessage: (message) => sent.push("warning: " + message),
      showInputBox: async () => fake.typed,
      registerTerminalLinkProvider: (provider) => { fake.provider = provider; return {dispose() {}}; },
    },
    workspace: {
      workspaceFolders: [{uri: {fsPath: "/work/my project"}}],
      getWorkspaceFolder: () => null,
      getConfiguration: () => ({get: (key) => ({command: "codehaven", startOllama: true, extraArguments: [], ...settings}[key])}),
    },
    commands: {registerCommand: (name, run) => { registered[name] = run; return {dispose() {}}; }},
    env: {openExternal() {}},
    Uri: {parse: (text) => text},
  };
  const original = Module._load;
  Module._load = function (name, ...rest) { return name === "vscode" ? fake : original.call(this, name, ...rest); };
  delete require.cache[require.resolve("../extension.js")];
  try { require("../extension.js").activate({subscriptions: []}); } finally { Module._load = original; }
  return fake;
}

test("every command in the manifest is registered", () => {
  const declared = require("../package.json").contributes.commands.map((entry) => entry.command).sort();
  assert.deepStrictEqual(Object.keys(load().registered).sort(), declared);
});

test("a question runs the codehaven command on the open folder, quoted as one argument", async () => {
  const fake = load();
  fake.typed = "What's in `store.py`?";
  await fake.registered["codehaven.ask"]();
  if (process.platform !== "win32") {
    assert.deepStrictEqual(fake.sent, [`'codehaven' '--start-ollama' --workspace '/work/my project' 'What'\\''s in \`store.py\`?'`]);
  }
});

test("settings change the command and its arguments", async () => {
  const fake = load({command: "/venv/bin/codehaven", startOllama: false, extraArguments: ["--model", "qwen2.5:14b"]});
  fake.typed = "Explain this project";
  await fake.registered["codehaven.ask"]();
  assert.match(fake.sent[0], /\/venv\/bin\/codehaven' '--model' 'qwen2\.5:14b' --workspace/);
  assert.doesNotMatch(fake.sent[0], /--start-ollama/);
});

test("an empty question runs nothing, and no open folder is explained", async () => {
  const fake = load();
  fake.typed = "   ";
  await fake.registered["codehaven.ask"]();
  assert.deepStrictEqual(fake.sent, []);
  fake.workspace.workspaceFolders = [];
  fake.typed = "Explain this project";
  await fake.registered["codehaven.ask"]();
  assert.match(fake.sent[0], /^warning: CodeHaven: open a folder first/);
});

test("citations in terminal output become links", () => {
  const links = load().provider.provideTerminalLinks({line: "- inventory/store.py:27-29 `Store.low_stock` Bug: uses <"});
  assert.deepStrictEqual(links.map((link) => [link.startIndex, link.length, link.citation.path, link.citation.start, link.citation.end]),
    [[2, 24, "inventory/store.py", 27, 29]]);
});
