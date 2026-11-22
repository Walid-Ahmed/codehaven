// CodeHaven for VS Code: ask about the open project, and jump from a cited path:line to the code.
//
// The extension holds no model and sends nothing anywhere. It runs the codehaven command that is installed on this
// computer, in a VS Code terminal or as the local chat page, and turns the citations CodeHaven prints into links
// that open the cited lines in the editor. CodeHaven itself is read-only, and so is this extension: it never edits a file.
"use strict";
const vscode = require("vscode");
const childProcess = require("child_process");
const path = require("path");
const {findCitations, parseCitation, shellQuote, pageLink} = require("./citations");

let pageProcess = null;  // The running `codehaven --gui`, when the chat page was opened from here.
let pageUrl = null;      // Its link, kept so a second "Open" shows the same page again.

// The folder questions are about: the workspace folder of the active file, else the first one. Undefined when none is open.
function projectFolder() {
  const active = vscode.window.activeTextEditor;
  const folder = (active && vscode.workspace.getWorkspaceFolder(active.document.uri)) || (vscode.workspace.workspaceFolders || [])[0];
  if (!folder) vscode.window.showWarningMessage("CodeHaven: open a folder first, so there is a project to ask about.");
  return folder ? folder.uri.fsPath : undefined;
}

// The command and the arguments every run gets, from the settings.
function baseCommand() {
  const settings = vscode.workspace.getConfiguration("codehaven");
  const extra = (settings.get("extraArguments") || []).map(String);
  return {command: settings.get("command") || "codehaven", extra: settings.get("startOllama") ? ["--start-ollama", ...extra] : extra};
}

// Run one question in the CodeHaven terminal. The answer's citations become links through the provider below.
function ask(question) {
  const folder = projectFolder();
  if (!folder || !question || !question.trim()) return;
  const {command, extra} = baseCommand();
  const windows = process.platform === "win32";
  const quote = (value) => shellQuote(value, windows);
  const terminal = vscode.window.terminals.find((found) => found.name === "CodeHaven") || vscode.window.createTerminal({name: "CodeHaven", cwd: folder});
  terminal.show(true);
  // PowerShell needs & in front of a quoted command; a POSIX shell runs it as written.
  terminal.sendText(`${windows ? "& " : ""}${quote(command)} ${extra.map(quote).join(" ")} --workspace ${quote(folder)} ${quote(question.trim())}`);
}

// The path of a file inside the project, with forward slashes, as CodeHaven names it; undefined for a file outside it.
function relativePath(uri) {
  const folder = vscode.workspace.getWorkspaceFolder(uri);
  return folder ? path.relative(folder.uri.fsPath, uri.fsPath).split(path.sep).join("/") : undefined;
}

// Open the cited lines and select them. The file is looked for in every workspace folder, and is never changed.
async function openCitation(citation) {
  for (const folder of vscode.workspace.workspaceFolders || []) {
    const uri = vscode.Uri.joinPath(folder.uri, ...citation.path.split("/"));
    try {
      await vscode.workspace.fs.stat(uri);
    } catch {
      continue;  // Not in this folder; try the next one.
    }
    const document = await vscode.workspace.openTextDocument(uri);
    const last = Math.min(citation.end, document.lineCount) - 1;
    const range = new vscode.Range(Math.min(citation.start, document.lineCount) - 1, 0, last, document.lineAt(last).text.length);
    const editor = await vscode.window.showTextDocument(document, {selection: range, preview: true});
    editor.revealRange(range, vscode.TextEditorRevealType.InCenterIfOutsideViewport);
    return;
  }
  vscode.window.showWarningMessage(`CodeHaven: ${citation.path} is not in an open folder.`);
}

// Start the local chat page for the project, or show it again if it is already running.
function openPage() {
  if (pageProcess && pageUrl) {
    vscode.env.openExternal(vscode.Uri.parse(pageUrl));
    return;
  }
  const folder = projectFolder();
  if (!folder) return;
  const {command, extra} = baseCommand();
  let output = "";
  const started = childProcess.spawn(command, ["--gui", "--no-browser", "--workspace", folder, ...extra], {cwd: folder});
  pageProcess = started;
  const read = (chunk) => {
    output = (output + chunk).slice(-8000);
    const link = !pageUrl && pageLink(output);
    if (link) {
      pageUrl = link;
      vscode.env.openExternal(vscode.Uri.parse(link));  // The link stays on this computer: it is a 127.0.0.1 address.
    }
  };
  started.stdout.on("data", read);
  started.stderr.on("data", read);
  started.on("error", (error) => {
    pageProcess = pageUrl = null;
    vscode.window.showErrorMessage(`CodeHaven could not be started (${error.message}). Set "codehaven.command" to the full path of the codehaven command.`);
  });
  started.on("exit", (code) => {
    // Only when it stopped before its page was ready is there something to explain; CodeHaven's own last lines say what.
    if (started === pageProcess && !pageUrl && code) {
      vscode.window.showErrorMessage("CodeHaven stopped before its page was ready: " + output.trim().split("\n").slice(-3).join(" "));
    }
    if (started === pageProcess) pageProcess = pageUrl = null;
  });
}

// Stop the chat page's server, if this window started one.
function stopPage() {
  if (pageProcess) pageProcess.kill();
  pageProcess = pageUrl = null;
}

// Called by VS Code when the extension starts: register the commands and the terminal links.
function activate(context) {
  const command = (name, run) => context.subscriptions.push(vscode.commands.registerCommand(name, run));
  command("codehaven.open", openPage);
  command("codehaven.stop", stopPage);
  command("codehaven.ask", async () => ask(await vscode.window.showInputBox({prompt: "Ask CodeHaven about this project", placeHolder: "Which function calls checkout_total?"})));
  command("codehaven.askAboutFile", () => {
    const editor = vscode.window.activeTextEditor;
    const file = editor && relativePath(editor.document.uri);
    if (file) ask(`Explain ${file}`);
  });
  command("codehaven.askAboutSelection", () => {
    const editor = vscode.window.activeTextEditor;
    const file = editor && relativePath(editor.document.uri);
    if (!file || editor.selection.isEmpty) return;
    // The question names the file and the lines, and CodeHaven reads them itself: the selected text is not pasted in.
    ask(`Explain lines ${editor.selection.start.line + 1}-${editor.selection.end.line + 1} of ${file}`);
  });
  command("codehaven.trace", () => {
    const editor = vscode.window.activeTextEditor;
    const range = editor && editor.document.getWordRangeAtPosition(editor.selection.active, /[A-Za-z_$][\w$]*/);
    if (range) ask(`Trace ${editor.document.getText(range)}`);
  });
  command("codehaven.goToCitation", async () => {
    const typed = await vscode.window.showInputBox({prompt: "A citation from a CodeHaven answer", placeHolder: "inventory/store.py:27-29"});
    const citation = typed && parseCitation(typed);
    if (citation) await openCitation(citation);
    else if (typed) vscode.window.showWarningMessage("CodeHaven: that is not a citation such as inventory/store.py:27-29.");
  });
  // Every path:line in any terminal becomes a link, so answers from the codehaven command can be followed with a click.
  context.subscriptions.push(vscode.window.registerTerminalLinkProvider({
    provideTerminalLinks: (terminalContext) => findCitations(terminalContext.line).map((citation) => (
      {startIndex: citation.index, length: citation.length, tooltip: "Open the cited lines", citation})),
    handleTerminalLink: (link) => openCitation(link.citation),
  }));
  context.subscriptions.push({dispose: stopPage});
}

// Called by VS Code when the extension stops: the chat page's server must not outlive the window.
function deactivate() {
  stopPage();
}

module.exports = {activate, deactivate};
