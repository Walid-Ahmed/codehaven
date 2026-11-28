// Runs inside a real VS Code, loaded by its own test runner (see run.sh): no packages are downloaded.
//
// It checks what the stand-in tests cannot: that VS Code activates the extension and registers its commands, that a
// citation opens the file with the cited lines selected, that a question goes to the CodeHaven terminal, and that the
// chat page really starts from the installed codehaven command and answers on its 127.0.0.1 link.
"use strict";
const assert = require("assert");
const http = require("http");
const path = require("path");
const vscode = require("vscode");

const COMMANDS = ["codehaven.open", "codehaven.stop", "codehaven.ask", "codehaven.askAboutFile",
                  "codehaven.askAboutSelection", "codehaven.trace", "codehaven.goToCitation"];

function get(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (response) => { response.resume(); response.on("end", () => resolve(response.statusCode)); }).on("error", reject);
  });
}

async function waitFor(check, seconds, what) {
  for (let tries = 0; tries < seconds * 10; tries++) {
    const value = await check();
    if (value) return value;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("Timed out waiting for " + what);
}

async function run() {
  const results = [];
  const step = async (name, body) => {
    await body();
    results.push("ok - " + name);
    console.log("ok - " + name);
  };
  const extension = vscode.extensions.all.find((found) => found.packageJSON.name === "codehaven");
  await step("the extension activates", async () => {
    assert.ok(extension, "extension not loaded");
    await extension.activate();
    assert.ok(extension.isActive);
  });
  await step("every command is registered", async () => {
    const known = await vscode.commands.getCommands(true);
    for (const command of COMMANDS) assert.ok(known.includes(command), command + " is missing");
  });
  const originalInput = vscode.window.showInputBox;
  await step("a typed citation opens the file with the lines selected", async () => {
    vscode.window.showInputBox = async () => "pkg/pricing.py:5-6";
    await vscode.commands.executeCommand("codehaven.goToCitation");
    const editor = vscode.window.activeTextEditor;
    assert.ok(editor, "no editor opened");
    assert.strictEqual(path.basename(editor.document.uri.fsPath), "pricing.py");
    assert.deepStrictEqual([editor.selection.start.line, editor.selection.end.line], [4, 5]);
    assert.ok(!editor.document.isDirty, "the file must not be changed");
  });
  await step("a question is sent to the CodeHaven terminal", async () => {
    vscode.window.showInputBox = async () => "Which function calls apply_discount?";
    await vscode.commands.executeCommand("codehaven.ask");
    await waitFor(() => vscode.window.terminals.find((terminal) => terminal.name === "CodeHaven"), 10, "the terminal");
  });
  vscode.window.showInputBox = originalInput;
  await step("the chat page starts from the installed command and answers on 127.0.0.1", async () => {
    const opened = [], errors = [];
    const originalOpen = vscode.env.openExternal, originalError = vscode.window.showErrorMessage;
    vscode.env.openExternal = async (uri) => { opened.push(uri.toString(true)); return true; };
    vscode.window.showErrorMessage = async (message) => { errors.push(message); };
    try {
      await vscode.commands.executeCommand("codehaven.open");
      const link = await waitFor(() => opened[0] || (errors[0] && Promise.reject(new Error(errors[0]))), 60,
                                 "the chat page link");
      assert.match(link, /^http:\/\/127\.0\.0\.1:\d+\/\?token=/);
      assert.strictEqual(await get(link), 200);
      await vscode.commands.executeCommand("codehaven.stop");
      await waitFor(async () => { try { await get(link); return false; } catch { return true; } }, 15, "the page to stop");
    } finally {
      vscode.env.openExternal = originalOpen;
      vscode.window.showErrorMessage = originalError;
    }
  });
  console.log(`${results.length} checks passed in VS Code ${vscode.version}`);
}

module.exports = {run};
