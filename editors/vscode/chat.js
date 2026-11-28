// The CodeHaven chat panel: questions and answers inside VS Code, with citations that open the cited lines.
//
// The panel talks to the same local chat page server as "Open the Chat Page", started from the installed codehaven
// command on 127.0.0.1 with its secret token. The extension makes the requests and passes the answer to the panel as
// it streams in; the panel itself reaches nothing. Nothing leaves this computer.
"use strict";
const http = require("http");

// POST one JSON request to the server and pass each line of its NDJSON reply to onEvent. Resolves at the end.
function request(link, path, body, onEvent) {
  const url = new URL(link);
  const token = url.searchParams.get("token");
  const data = Buffer.from(JSON.stringify(body));
  return new Promise((resolve, reject) => {
    const sent = http.request({host: "127.0.0.1", port: url.port, path, method: "POST",
                               headers: {"Content-Type": "application/json", "Content-Length": data.length,
                                         "X-CodeHaven-Token": token, "Host": `127.0.0.1:${url.port}`}}, (response) => {
      let buffer = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => {
        buffer += chunk;
        let end;
        while ((end = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, end).trim();
          buffer = buffer.slice(end + 1);
          if (line) onEvent(JSON.parse(line), response.statusCode);
        }
      });
      response.on("end", () => {
        if (buffer.trim()) onEvent(JSON.parse(buffer), response.statusCode);
        resolve();
      });
    });
    sent.on("error", reject);
    sent.end(data);
  });
}

// The panel's page. Its script only renders messages and posts the user's questions and clicks to the extension.
function html(nonce, cspSource) {
  return `<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>
  body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 0 8px; }
  #messages { display: flex; flex-direction: column; gap: 10px; padding-bottom: 90px; }
  .q { align-self: flex-end; background: var(--vscode-input-background); padding: 6px 10px; border-radius: 8px; white-space: pre-wrap; }
  .a { white-space: pre-wrap; line-height: 1.45; }
  .note { opacity: 0.7; font-size: 12px; }
  .warn { color: var(--vscode-editorWarning-foreground); font-size: 12px; }
  a.cite { color: var(--vscode-textLink-foreground); cursor: pointer; }
  pre.diff { background: var(--vscode-textCodeBlock-background); padding: 6px; overflow: auto; font-size: 12px; }
  #box { position: fixed; bottom: 0; left: 0; right: 0; padding: 8px; background: var(--vscode-sideBar-background); display: flex; gap: 6px; }
  textarea { flex: 1; resize: none; height: 44px; color: var(--vscode-input-foreground); background: var(--vscode-input-background); border: 1px solid var(--vscode-input-border, transparent); }
  button { color: var(--vscode-button-foreground); background: var(--vscode-button-background); border: none; padding: 4px 10px; cursor: pointer; }
</style></head><body>
<div id="messages"><div class="note">Ask about this project. Answers cite the lines they rely on; click one to open it. Type /help for commands.</div></div>
<div id="box"><textarea id="question" placeholder="Which function calls checkout_total?"></textarea><button id="send">Ask</button></div>
<script nonce="${nonce}">
  const vscode = acquireVsCodeApi();
  const messages = document.getElementById("messages"), question = document.getElementById("question");
  let current = null, busy = false;
  const CITE = /(?<![\\w./-])((?:repo\\/)?(?:[\\w.@+-]+\\/)*[\\w.@+-]*[A-Za-z_][\\w@+-]*\\.[A-Za-z]\\w{0,7}):(\\d{1,7})(?:-(\\d{1,7}))?(?![\\w/])/g;
  function add(kind, text) { const element = document.createElement("div"); element.className = kind; element.textContent = text || ""; messages.append(element); element.scrollIntoView(); return element; }
  function render(element, text) {
    element.replaceChildren();
    let last = 0;
    for (const match of text.matchAll(CITE)) {
      element.append(text.slice(last, match.index));
      const link = Object.assign(document.createElement("a"), {className: "cite", textContent: match[0]});
      link.addEventListener("click", () => vscode.postMessage({type: "open", path: match[1].replace(/^repo\\//, ""), start: Number(match[2]), end: Number(match[3] || match[2])}));
      element.append(link);
      last = match.index + match[0].length;
    }
    element.append(text.slice(last));
  }
  function ask(text) {
    if (!text.trim() || busy) return;
    vscode.postMessage({type: "ask", question: text});  // The extension shows it, then the answer.
  }
  document.getElementById("send").addEventListener("click", () => { ask(question.value); question.value = ""; });
  question.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); ask(question.value); question.value = ""; } });
  vscode.postMessage({type: "ready"});
  window.addEventListener("message", ({data}) => {
    if (data.type === "question") { busy = true; add("q", data.question); current = add("a", "…"); current.dataset.text = ""; return; }
    if (!current) return;
    if (data.type === "text") { current.dataset.text += data.text; render(current, current.dataset.text); }
    else if (data.type === "diff") {
      const box = add("a", "Proposed change: " + data.summary);
      const pre = document.createElement("pre"); pre.className = "diff"; pre.textContent = data.diff; box.append(pre);
      for (const [label, apply] of [["Apply", true], ["Decline", false]]) {
        const button = Object.assign(document.createElement("button"), {textContent: label});
        button.addEventListener("click", () => { vscode.postMessage({type: "decide", apply}); box.querySelectorAll("button").forEach((b) => b.disabled = true); });
        box.append(button, " ");
      }
    } else if (data.type === "done") {
      render(current, data.answer || current.dataset.text || "(no answer)");
      if (data.issues && data.issues.length) add("warn", "Unverified: check the cited lines. " + data.issues.join(" "));
      busy = false;
    } else if (data.type === "error" || data.type === "stopped") {
      current.textContent = data.type === "stopped" ? "Stopped." : data.message; current.className = "warn"; busy = false;
    }
  });
</script></body></html>`;
}

module.exports = {request, html};
