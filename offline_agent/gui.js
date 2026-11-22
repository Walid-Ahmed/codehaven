"use strict";
// ---- Setup: shared state, limits, and suggested questions.
// The secret token from the link the server printed; every request sends it back, or the server refuses.
const TOKEN = new URLSearchParams(location.search).get("token") || "";
const $ = (id) => document.getElementById(id);
// busy: a request is running; indexed: something is open; uploaded: it is one uploaded file; browsing: the folder browser's folder.
// stoppable: the running request is an answer or flashcards, which Stop can end; stopping: Stop was clicked.
let busy = false, indexed = false, uploaded = false, browsing = "", stoppable = false, stopping = false;
// indexing: "index" or "clone" while a repository is being indexed or downloaded; that button then reads Stop.
let indexing = "";
// The folder of the open repository, for links that open a cited file in an editor; null for an uploaded file.
let repoRoot = null;
const UPLOAD_BYTES = 64000;  // The same limit as every file CodeHaven reads.
const PICTURE_BYTES = 8000000;  // A picture of code is read into text by Tesseract on this computer.
const isDocument = (file) => /\.(ipynb|pdf)$/i.test(file.name) || file.type === "application/pdf";
const isPicture = (file) => /^image\/(png|jpeg)$/.test(file.type) || /\.(png|jpe?g)$/i.test(file.name);
// ---- Language: English, or Arabic chosen with the button in the top bar and remembered in this browser only.
// The Arabic page is the same page: its English text is the key each Arabic sentence is looked up by.
let LANG = "en";
try { if (localStorage.getItem("codehaven.language") === "ar") LANG = "ar"; } catch {}
// The Arabic sentences, looked up by their English ones, are kept in gui_arabic.js and put here when the page is served.
/*__ARABIC__*/
// A sentence in the page's language, with {name} places filled from values.
function t(text, values) {
  const found = LANG === "ar" && ARABIC[text] || text;
  return values ? found.replace(/\{(\w+)\}/g, (place, name) => values[name] ?? place) : found;
}
// A count with its noun: "3 files". Arabic has its own form for one, for two, for 3 to 10, and for 11 or more.
function count(number, word) {
  if (LANG !== "ar") return plural(number, word);
  const forms = ARABIC_NOUNS[word];
  if (number === 1 || number === 2) return forms[number - 1];
  return `${number} ${number % 100 >= 3 && number % 100 <= 10 ? forms[2] : forms[3]}`;
}
// Put the page that was written in English into Arabic, and lay it out right to left. Runs once, before anything
// else reads text from the page.
function translatePage() {
  document.documentElement.lang = "ar"; document.documentElement.dir = "rtl";
  const texts = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let node; (node = texts.nextNode());) {
    const text = node.nodeValue.trim();
    if (text && ARABIC[text] && !node.parentElement.closest("script, style")) node.nodeValue = node.nodeValue.replace(text, ARABIC[text]);
  }
  for (const element of document.querySelectorAll("[title], [placeholder], [aria-label], [data-drop]")) {
    for (const name of ["title", "placeholder", "aria-label", "data-drop"]) {
      const value = element.getAttribute(name);
      if (value && ARABIC[value]) element.setAttribute(name, ARABIC[value]);
    }
  }
  // The button offers the other language, written in that language.
  Object.assign($("language"), {textContent: "English", title: "Show this page in English", lang: "en"});
}
if (LANG === "ar") translatePage();
$("language").addEventListener("click", () => {
  if (busy) return;
  // The empty state is hidden once a chat has messages; only then is there something to lose.
  if ($("empty").hidden && !confirm(t("Switching the language reloads the page, which starts a new chat. Continue?"))) return;
  try { localStorage.setItem("codehaven.language", LANG === "ar" ? "en" : "ar"); } catch { toast(t("This browser did not allow saving the language.")); return; }
  location.reload();
});
// Suggested questions, as [button label, question] pairs. The Arabic ones are asked, and so answered, in Arabic.
const REPOSITORY_SUGGESTIONS = LANG === "ar" ? [
  ["اشرح المشروع", "اشرح هذا المشروع"],
  ["اشرح README.md", "اشرح README.md"],
  ["الوحدات الرئيسية", "ما هي الوحدات الرئيسية وما وظيفة كل منها؟"],
  ["الاختبارات", "ما هي الاختبارات الموجودة وماذا تفحص؟"],
  ["جولة في المشروع", "أعطني جولة في المشروع: أي الملفات يجب أن أقرأ أولا؟"],
  ["فجوات الاختبارات", "أي الدوال بدون اختبارات؟"],
] : [
  ["Explain this project", "Explain this project"],
  ["Explain README.md", "Explain README.md"],
  ["Main modules", "What are the main modules and what does each do?"],
  ["Tests", "Which tests are there and what do they check?"],
  ["Project tour", "Give me a project tour: which files should I read first?"],
  ["Test gaps", "Which functions have no tests?"],
];
const GIT_SUGGESTIONS = LANG === "ar" ? [
  ["تغييراتي", "اشرح تغييراتي"],
  ["آخر commit", "ماذا تغير في آخر commit؟"],
] : [
  ["My changes", "Explain my uncommitted changes"],
  ["Last commit", "What changed in my last commit?"],
];
// A short message that goes away by itself, for errors that do not belong to one part of the page.
let toastTimer = null;
// Show a short message at the bottom of the page for a few seconds.
function toast(message) {
  const box = $("toast");
  box.textContent = message; box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { box.hidden = true; }, 6000);
}
// Search by meaning uses a second local model; say whether it is on, and how far along it is.
function meaningNote(meaning) {
  if (!meaning) return "";
  if (meaning.failed) return " " + t("Search by meaning stopped, so answers use key words.");
  return " " + (meaning.ready >= meaning.total ? t("Search by meaning is on.")
    : t("Search by meaning: {ready} of {total} passages ready, the rest are being prepared.", meaning));
}
// Suggested questions for one uploaded file, as [button label, question] pairs.
function fileSuggestions(name) {
  return LANG === "ar" ? [
    [`اشرح ${name}`, `اشرح ${name}`],
    ["الأجزاء الرئيسية", `ما هي الدوال أو الأقسام الرئيسية في ${name} وما وظيفة كل منها؟`],
    ["مراجعة", `راجع ${name}`],
  ] : [
    [`Explain ${name}`, `Explain ${name}`],
    ["Main parts", `What are the main functions or sections in ${name}, and what does each do?`],
    ["Review", `Review ${name}`],
  ];
}
// Show the suggestions as buttons; clicking one asks its question.
function setSuggestions(items) {
  $("suggestions").replaceChildren(...items.map(([label, question]) => {
    const button = document.createElement("button");
    button.textContent = label;
    button.addEventListener("click", () => ask(question));
    return button;
  }));
}

// ---- Talking to the server.
// One JSON request. A refused one throws the server's own error message.
async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {"Content-Type": "application/json", "X-CodeHaven-Token": TOKEN}});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

// ---- Streaming answers from the server.
// Events of a streamed reply: the server sends one JSON object per line.
async function* streamEvents(path, body) {
  const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json", "X-CodeHaven-Token": TOKEN},
                                      body: JSON.stringify(body)});
  if (!response.ok) { const data = await response.json().catch(() => ({})); throw new Error(data.error || response.statusText); }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const raw = buffer.slice(0, newline); buffer = buffer.slice(newline + 1);
      if (raw.trim()) yield JSON.parse(raw);
    }
  }
}

// ---- Safe rendering: escape everything first, then add a small set of markup.
function esc(text) {
  return text.replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}
// A citation such as src/app.py:12 or src/app.py:12-20, with or without repo/ in front.
const CITE = /(^|[\s(\[>,;:،؛])((?:repo\/)?[\wء-ي./-]+\.[A-Za-z]\w*):(\d+)(?:-(\d+))?(?![\w/])/g;
// Arabic and Hebrew read right to left. Code and file names are left out when deciding, so an Arabic sentence
// that starts with store.py:12 or `add` is still laid out right to left.
const LETTER = /[A-Za-z֐-ࣿיִ-﷿ﹰ-ﻼ]/;
// "rtl" when the first real letter of text is Arabic or Hebrew, otherwise "ltr". Code and file names are skipped.
function direction(text) {
  const first = text.replace(/`[^`]*`/g, " ").replace(/\S*[\w)]\.\w+(?::\d+(?:-\d+)?)?/g, " ").match(LETTER);
  return first && !/[A-Za-z]/.test(first[0]) ? "rtl" : "ltr";
}
const dirAttr = (text) => ` dir="${direction(text)}"`;
// One line of text as safe HTML: code spans, bold, and clickable citations.
function inline(text) {
  let html = esc(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return html.replace(CITE, (m, pre, path, start, end) =>
    `${pre}<a href="#" class="cite" data-path="${path}" data-start="${start}" data-end="${end || start}">${path}:${start}${end ? "-" + end : ""}</a>`);
}
// A whole answer as safe HTML: paragraphs, lists, headings, and fenced code blocks.
function render(text) {
  const out = [];
  let list = null, para = [];
  const flushPara = () => { if (para.length) { out.push(`<p${dirAttr(para.join(" "))}>${para.map(inline).join("<br>")}</p>`); para = []; } };
  const flushList = () => { if (list) { out.push(`<${list.tag}${dirAttr(list.first)}>${list.items.join("")}</${list.tag}>`); list = null; } };
  const lines = text.replace(/\r/g, "").split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      flushPara(); flushList();
      const code = [];
      while (++i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i]);
      out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
      continue;
    }
    const bullet = line.match(/^\s*(?:[-*•]|(\d+)[.)])\s+(.*)$/);
    if (bullet) {
      flushPara();
      const tag = bullet[1] ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = {tag, items: [], first: bullet[2]}; }
      list.items.push(`<li>${inline(bullet[2])}</li>`);
      continue;
    }
    const heading = line.match(/^\s*#{1,6}\s+(.*)$/);
    if (heading) { flushPara(); flushList(); out.push(`<h4${dirAttr(heading[1])}>${inline(heading[1])}</h4>`); continue; }
    if (!line.trim()) { flushPara(); flushList(); continue; }
    if (list && /^\s{2,}\S/.test(line)) { list.items[list.items.length - 1] = list.items[list.items.length - 1].replace(/<\/li>$/, `<br>${inline(line.trim())}</li>`); continue; }
    flushList();
    para.push(line);
  }
  flushPara(); flushList();
  return out.join("");
}

// ---- Recent repositories live only in this browser.
function recent() { try { return JSON.parse(localStorage.getItem("codehaven.recent") || "[]"); } catch { return []; } }
// Put a repository at the top of the recent list (six at most), kept in this browser only.
function remember(path) {
  try { localStorage.setItem("codehaven.recent", JSON.stringify([path, ...recent().filter((p) => p !== path)].slice(0, 6))); } catch {}
  showRecent();
}
// Draw the recent repositories as buttons that reopen them.
function showRecent() {
  const paths = recent();
  $("recent-section").hidden = !paths.length;
  $("recent").replaceChildren(...paths.map((path) => {
    const button = document.createElement("button");
    button.textContent = path.split("/").filter(Boolean).pop() || path;
    button.title = path;
    button.addEventListener("click", () => { $("path").value = path; indexRepository(); });
    return button;
  }));
}

// ---- Indexing.
const UPLOAD_HINT = $("upload-status").textContent;
// Update the whole page from the server's summary of what is open.
function setIndexed(summary) {
  indexed = Boolean(summary.repository || summary.upload);
  repoRoot = summary.repository || null;
  uploaded = Boolean(summary.upload);
  $("model-name").textContent = summary.model || t("a local model");
  setClone(Boolean(summary.clone));
  setSaved(summary);
  setChats(summary.chats);
  if (!indexed) return;
  $("start").hidden = true; $("suggest-label").hidden = false;
  $("empty-title").textContent = t("Ask about your code");
  document.body.classList.remove("menu-open"); $("overlay").classList.remove("open");
  $("repo-label").classList.remove("empty");
  $("index-pill").hidden = false;
  const status = $("index-status"), upload = $("upload-status");
  status.classList.remove("error"); upload.classList.remove("error");
  if (summary.upload) {
    $("repo-label").textContent = t("{name} · uploaded file", {name: summary.upload});
    $("repo-label").title = "";
    $("index-pill").textContent = count(1, "file");
    status.textContent = t("Choose a folder, then index it to chat about a whole repository.");
    upload.replaceChildren(t("Chatting about "), Object.assign(document.createElement("strong"), {textContent: summary.upload}),
      t(". The copy CodeHaven reads is deleted when you open something else; your original is never touched."));
    $("empty-text").textContent = t("Ask how something in {name} works. Answers cite the exact lines they rely on; click a citation to see the code.", {name: summary.upload});
    setSuggestions(fileSuggestions(summary.upload));
    setPicture(summary);
  } else {
    setPicture({});
    $("repo-label").textContent = summary.repository.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || summary.repository;
    $("repo-label").title = summary.repository;
    $("index-pill").textContent = t("{files} indexed", {files: count(summary.files, "file")});
    status.innerHTML = t("<strong>{n}</strong> files indexed.", {n: Number(summary.files)}) +
      (summary.limited ? " " + t("Index limits were reached, so some files are left out; answers can still search and read them.") : "") +
      meaningNote(summary.meaning);
    upload.textContent = UPLOAD_HINT;
    $("path").value = summary.repository;
    $("empty-text").textContent = t("Ask how something in this repository works. Answers cite the exact lines they rely on; click a citation to see the code.");
    setSuggestions(REPOSITORY_SUGGESTIONS.concat(summary.git ? GIT_SUGGESTIONS : []));
  }
  $("suggestions").hidden = false;
  updateComposer();
}
// Index the folder typed or chosen in the sidebar, and start an empty chat about it.
async function indexRepository() {
  if (indexing === "index") return stopIndexing();
  const path = $("path").value.trim();
  if (!path || busy) return;
  busy = true; indexing = "index"; updateComposer();
  const status = $("index-status");
  status.classList.remove("error");
  status.textContent = t("Indexing… this usually takes a few seconds.");
  try {
    const summary = await api("/api/open", {method: "POST", body: JSON.stringify({path})});
    if (summary.stopped) {
      status.textContent = t(indexed ? "Stopped. Nothing new was indexed; the project open before is still open."
        : "Stopped. Nothing was indexed.");
      return;
    }
    clearChat(); resetStudy();
    setIndexed(summary);
    remember(summary.repository);
  } catch (error) {
    status.classList.add("error");
    status.textContent = error.message;
  } finally {
    busy = stopping = false; indexing = ""; updateComposer();
  }
}
// Stop the index or download in progress; the request that started it then returns with "stopped".
async function stopIndexing() {
  if (!indexing || stopping) return;
  stopping = true; updateComposer();
  try { await api("/api/stop", {method: "POST", body: "{}"}); }
  catch (error) { stopping = false; updateComposer(); toast(error.message); }
}

// ---- Long-term memory, unless the server was started with --no-remember: checked answers kept on this computer.
let remembering = false, savedCount = 0;
// Show how many answers are saved. kept is true when the answer just given was added.
function setSaved(summary, kept) {
  remembering = Boolean(summary.remember ?? remembering);
  savedCount = summary.saved || 0;
  $("saved-section").hidden = !remembering;
  if (!remembering) return;
  $("privacy-saved").textContent = t(" Checked answers and your chats are saved on this computer only.");
  $("saved-status").textContent = uploaded ? t("Uploaded files are never remembered.")
    : !indexed ? t("Checked answers and chats are kept on this computer.")
    : t("{answers} for this project{added}. An answer is used again only while the files it cites are unchanged.",
        {answers: count(savedCount, "saved answer"), added: kept ? t(" (one just added)") : ""});
  $("forget").disabled = busy || uploaded || !(savedCount || chatCount);
}
$("forget").addEventListener("click", async () => {
  if (!confirm(t("Delete the {answers} and {chats} for this project? This cannot be undone.",
                 {answers: count(savedCount, "saved answer"), chats: count(chatCount, "saved chat")}))) return;
  try { const summary = await api("/api/forget", {method: "POST", body: "{}"}); setSaved(summary); setChats(summary.chats); } catch (error) { toast(error.message); }
});

// ---- Saved chats: earlier conversations about this repository, kept on this computer and reopened with a click.
let chatCount = 0;
// Rebuild the sidebar's list of saved chats, each with a delete button.
function setChats(chats) {
  chats = chats || [];
  chatCount = chats.length;
  $("forget").disabled = busy || uploaded || !(savedCount || chatCount);
  $("chats-section").hidden = !chats.length || uploaded;
  $("chats").replaceChildren(...chats.map((chat) => {
    const row = document.createElement("div"); row.className = "chat-row";
    const open = document.createElement("button");
    open.textContent = chat.title; open.dir = "auto";
    open.title = t("{questions} · last used {when}", {questions: count(chat.questions, "question"), when: chat.updated.replace("T", " ")});
    open.addEventListener("click", () => openChat(chat.id));
    const remove = document.createElement("button"); remove.className = "chat-delete"; remove.textContent = "×";
    remove.title = t("Delete this saved chat"); remove.setAttribute("aria-label", t("Delete the saved chat {title}", chat));
    remove.addEventListener("click", async () => {
      if (busy || !confirm(t("Delete the saved chat \"{title}\"? This cannot be undone.", chat))) return;
      try { setChats((await api("/api/chat/delete", {method: "POST", body: JSON.stringify({id: chat.id})})).chats); } catch (error) { toast(error.message); }
    });
    row.append(open, remove);
    return row;
  }));
}
// Reopen a saved chat: its questions and answers are shown again, and the server remembers them for follow-ups.
async function openChat(id) {
  if (busy) return;
  try {
    const chat = await api("/api/chat", {method: "POST", body: JSON.stringify({id})});
    setView("chat");
    clearChat();
    addMessage("note").textContent = t("Saved chat, last used {when}. The code may have changed since: cited lines open the files as they are now.", {when: chat.updated.replace("T", " ")});
    for (const turn of chat.turns) {
      keepTurn(turn.question, turn.answer, turn.issues);
      addMessage("user").textContent = turn.question;
      const card = addMessage("assistant");
      const answer = document.createElement("div"); answer.className = "answer"; answer.innerHTML = render(turn.answer);
      card.append(answer);
      if (turn.issues.length) {
        const checks = document.createElement("div"); checks.className = "checks";
        const title = document.createElement("strong"); title.textContent = t("Unverified: check the cited lines before relying on this answer.");
        const list = document.createElement("ul");
        for (const issue of turn.issues) { const item = document.createElement("li"); item.dir = "auto"; item.textContent = issue; list.append(item); }
        checks.append(title, list); card.append(checks);
      }
    }
    $("memory-status").textContent = t("Reopened a saved chat. Follow-ups remember {remembered} of its questions.", chat);
    scrollDown();
  } catch (error) { toast(error.message); }
}

// ---- GitHub, only when the server was started with --allow-clone: git on this computer downloads it once.
let cloneOn = false;
// Show the GitHub section and change the offline wording, the first time the server says cloning is on.
function setClone(on) {
  if (on === cloneOn) return;
  cloneOn = on;
  $("clone-section").hidden = !on;
  if (!on) return;
  $("offline-pill").textContent = t("Offline · GitHub download on");
  $("offline-pill").title = t("CodeHaven connects to the internet only to download a GitHub repository you ask for. Nothing is uploaded.");
  $("offline-text").textContent = t("Your code never leaves this computer. GitHub downloads are on: they run only when you click Download, and upload nothing.");
  $("privacy-network").textContent = t("Only GitHub downloads you ask for use the internet; nothing is sent.");
}
// Download a GitHub repository by its address, then index the copy. Only offered with --allow-clone.
async function cloneRepository() {
  if (indexing === "clone") return stopIndexing();
  const address = $("github").value.trim();
  if (!address || busy) return;
  busy = true; indexing = "clone"; updateComposer();
  const status = $("clone-status");
  status.classList.remove("error");
  status.textContent = t("Downloading from GitHub… this can take a minute for large repositories.");
  try {
    const summary = await api("/api/clone", {method: "POST", body: JSON.stringify({address})});
    if (summary.stopped) {
      status.textContent = t("Stopped. Nothing was downloaded or kept.");
      return;
    }
    clearChat(); resetStudy();
    setIndexed(summary);
    remember(summary.repository);
    status.replaceChildren(summary.cloned ? t("Downloaded ") : t("Opened the copy already on this computer of "),
      Object.assign(document.createElement("strong"), {textContent: summary.source}),
      summary.cloned ? t(". It is kept on this computer and opens offline from Recent.") : t(", without using the internet."));
    $("github").value = "";
  } catch (error) {
    status.classList.add("error");
    status.textContent = error.message;
  } finally {
    busy = stopping = false; indexing = ""; updateComposer();
  }
}
$("clone").addEventListener("click", cloneRepository);
$("github").addEventListener("keydown", (event) => { if (event.key === "Enter") cloneRepository(); });

// ---- A picture of code: Tesseract on this computer read it into the uploaded file. Reading is never perfect,
// so the lines to compare with the picture are listed, and the text can be saved as a file.
function lineLink(name, number, text) {  // Opened by the same click handler as the citations in answers.
  const link = Object.assign(document.createElement("a"), {href: "#", className: "cite", textContent: text || String(number)});
  link.dataset.path = name; link.dataset.start = link.dataset.end = number;
  return link;
}
// What to say about a picture that was read: how many lines, which to compare, and whether it is valid Python.
function pictureNotes(summary) {
  const picture = summary.picture, name = summary.upload;
  const notes = [t("Read {lines} from the picture {image} into ", {lines: count(picture.lines, "line"), image: picture.image}), Object.assign(document.createElement("strong"), {textContent: name}),
    t(". Reading a picture can get characters wrong, so check the text before you run it.")];
  if (picture.check.length) notes.push(t(" Compare with the picture: line "), ...picture.check.flatMap((number, index) => [index ? ", " : "", lineLink(name, number)]), ".");
  if (picture.syntax) notes.push(Object.assign(document.createElement("strong"), {textContent: t(" Not valid Python yet ({problem}).", {problem: picture.syntax})}));
  else if (name.endsWith(".py")) notes.push(t(" It is valid Python."));
  return notes;
}
// The sidebar note and Save button for a picture, notebook, or PDF that was read into text.
function setPicture(summary) {
  $("picture-status").hidden = $("save-row").hidden = !(summary.picture || summary.document);
  if (!summary.picture && !summary.document) return;
  $("picture-status").replaceChildren(...(summary.picture ? pictureNotes(summary) : [summary.document.note]));
  $("save-text").textContent = t("Save {name}…", {name: summary.upload});
}
// Save the text read from the upload as a file, through a download link that exists only for the click.
async function saveText(name) {
  try {
    const file = await api("/api/text");
    if (name && file.name !== name) throw new Error(t("{name} is no longer open, so it cannot be saved. Attach the picture again.", {name}));
    const link = Object.assign(document.createElement("a"), {href: URL.createObjectURL(new Blob([file.text], {type: "text/plain"})), download: file.name});
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  } catch (error) { toast(error.message); }
}
$("save-text").addEventListener("click", () => saveText());
// The note a file leaves in the chat when it is attached, pasted, or uploaded while a conversation is open.
function uploadNote(summary, chatting) {
  const name = summary.upload;
  const notes = summary.picture ? pictureNotes(summary)
    : [summary.document ? summary.document.note + " " : "", t("Now chatting about "), Object.assign(document.createElement("strong"), {textContent: name}), "."];
  if (chatting) notes.push(t(" The chat above is kept and follow-up questions still remember it, but its citations point to what was open before, so they no longer open."));
  if (summary.picture || summary.document) {
    const save = Object.assign(document.createElement("button"), {textContent: t("Save {name}…", {name})});
    save.addEventListener("click", () => saveText(name));
    notes.push(" ", lineLink(name, 1, t("View {name}", {name})), save);
  }
  return notes;
}

// ---- One uploaded file, for a quick question without a repository. The file goes only to this local server.
// The chat is kept: a file attached or pasted in the composer (fromChat) reports its progress there as well.
async function uploadFile(file, fromChat = false) {
  if (!file || busy) return;
  const status = $("upload-status"), picture = isPicture(file) || isDocument(file), chatting = $("empty").hidden;
  let note = null;
  const report = (text, failed = false) => {
    status.classList.toggle("error", failed); status.textContent = text;
    if (!fromChat) return;
    note = note || addMessage("note");
    note.classList.toggle("error", failed); note.textContent = text; scrollDown();
  };
  if (file.size > (picture ? PICTURE_BYTES : UPLOAD_BYTES)) {
    const size = {name: file.name, size: Math.ceil(file.size / (picture ? 1000000 : 1000))};
    report(isDocument(file) ? t("{name} is {size} MB; CodeHaven reads notebooks and PDFs up to 8 MB.", size)
      : picture ? t("{name} is {size} MB; CodeHaven reads pictures up to 8 MB. Crop it to the code.", size)
      : t("{name} is {size} KB; CodeHaven reads text files up to 64 KB. Index its folder instead to ask about part of it.", size), true);
    return;
  }
  busy = true; updateComposer();
  report(t(isDocument(file) ? "Reading {name} into text…" : picture ? "Reading the code in {name}… this takes a few seconds." : "Reading {name}…", {name: file.name}));
  try {
    const response = await fetch("/api/upload?name=" + encodeURIComponent(file.name),
                                 {method: "POST", headers: {"Content-Type": "application/octet-stream", "X-CodeHaven-Token": TOKEN}, body: file});
    const summary = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(summary.error || response.statusText);
    resetStudy();
    setIndexed(summary);
    if (summary.picture || chatting || note) {
      note = note || addMessage("note");
      note.replaceChildren(...uploadNote(summary, chatting)); scrollDown();
    }
  } catch (error) {
    report(error.message, true);
  } finally {
    busy = false; updateComposer();
  }
}
let attaching = false;  // Whether the file chooser was opened from the composer rather than the sidebar.
$("upload").addEventListener("click", () => { attaching = false; $("file-input").click(); });
$("attach").addEventListener("click", () => { attaching = true; $("file-input").click(); });
$("file-input").addEventListener("change", () => { const file = $("file-input").files[0]; $("file-input").value = ""; uploadFile(file, attaching); });
// A copied screenshot or file pasted anywhere on the page is attached. Pasted text is left alone, including text
// copied from an app that also puts a picture of it on the clipboard.
document.addEventListener("paste", (event) => {
  const data = event.clipboardData;
  if (!data || !data.files.length || data.getData("text/plain")) return;
  event.preventDefault();
  uploadFile(data.files[0], true);
});
// Dragging a file over the page. dragenter and dragleave fire for every element crossed, so a count tells when the
// file has really left the page.
let dragDepth = 0;
const dragsFiles = (event) => Array.from(event.dataTransfer?.types || []).includes("Files");
document.addEventListener("dragenter", (event) => { if (!dragsFiles(event)) return; event.preventDefault(); dragDepth++; if (!busy) document.body.classList.add("dropping"); });
document.addEventListener("dragover", (event) => { if (dragsFiles(event)) event.preventDefault(); });
document.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dropping"); } });
document.addEventListener("drop", (event) => {
  if (!dragsFiles(event)) return;
  event.preventDefault();
  dragDepth = 0; document.body.classList.remove("dropping");
  const files = event.dataTransfer.files;
  if (files.length > 1) { $("upload-status").classList.add("error"); $("upload-status").textContent = t("Drop one file at a time, or index the folder that holds them."); return; }
  uploadFile(files[0], true);
});

// ---- Folder browser.
// List the folders inside path in the folder browser; clicking one goes into it.
async function browse(path) {
  try {
    const data = await api("/api/dirs?path=" + encodeURIComponent(path || ""));
    browsing = data.path;
    $("browser-path").textContent = data.path + (data.repository ? t("  (Git repository)") : "");
    $("up").disabled = !data.parent;
    $("up").onclick = () => browse(data.parent);
    const items = data.entries.map((entry) => {
      const button = document.createElement("button");
      const name = document.createElement("span");
      name.textContent = "📁 " + entry.name;
      button.append(name);
      if (entry.repository) { const tag = document.createElement("span"); tag.className = "tag"; tag.textContent = t("Git repository"); button.append(tag); }
      button.addEventListener("click", () => browse(entry.path));
      return button;
    });
    if (!items.length) { const empty = document.createElement("div"); empty.className = "index-status"; empty.style.padding = "8px 10px"; empty.textContent = t("No folders here."); items.push(empty); }
    $("browser-list").replaceChildren(...items);
  } catch (error) {
    $("browser-path").textContent = error.message;
  }
}
// Opening and closing the dialogs, the drawer, and (on a small screen) the sidebar.
function openBrowser() { document.body.classList.remove("menu-open"); $("overlay").classList.add("open"); $("browser").classList.add("open"); browse($("path").value.trim()); }
// Close whatever is open on top of the page: the menu, the folder browser, the file picker, and the code drawer.
function closeOverlays() { document.body.classList.remove("menu-open"); for (const id of ["overlay", "browser", "picker", "drawer"]) $(id).classList.remove("open"); }
$("menu").addEventListener("click", () => { document.body.classList.add("menu-open"); $("overlay").classList.add("open"); });
$("start-folder").addEventListener("click", openBrowser);
$("start-file").addEventListener("click", () => { attaching = false; $("file-input").click(); });

// ---- Cited code.
// A link VS Code understands on its own, with no extension: vscode://file/<full path>:<line>.
function editorLink(path, line) {
  const full = (repoRoot + "/" + path).replace(/\\/g, "/").replace(/\/{2,}/g, "/");
  return "vscode://file" + (full.startsWith("/") ? "" : "/") + encodeURI(full) + ":" + line;
}
// Open the drawer on a file with the cited lines highlighted and a few lines around them.
async function showCitation(path, start, end) {
  $("drawer-editor").hidden = true;
  $("drawer-title").textContent = `${path}:${start}${end !== start ? "-" + end : ""}`;
  $("drawer-code").textContent = t("Loading…");
  $("overlay").classList.add("open"); $("drawer").classList.add("open");
  try {
    const data = await api(`/api/file?path=${encodeURIComponent(path)}&start=${start}&end=${end}`);
    $("drawer-title").textContent = `${data.path}:${data.start}${data.end !== data.start ? "-" + data.end : ""}  ·  ${LANG === "ar" ? t("{lines} in the file", {lines: data.total}) : data.total + " lines"}`;
    if (repoRoot) { $("drawer-editor").href = editorLink(data.path, data.start); $("drawer-editor").hidden = false; }
    $("drawer-code").replaceChildren(...data.lines.map((text, index) => {
      const number = data.first + index;
      const row = document.createElement("div");
      row.className = "line" + (number >= data.start && number <= data.end ? " hit" : "");
      const n = document.createElement("span"); n.className = "n"; n.textContent = number;
      const code = document.createElement("span"); code.textContent = text || " ";
      row.append(n, code);
      return row;
    }));
    const hit = $("drawer-code").querySelector(".hit");
    if (hit) hit.scrollIntoView({block: "center"});
  } catch (error) {
    $("drawer-code").textContent = error.message;
  }
}
// A click on any citation link, wherever it is on the page, opens the drawer.
document.addEventListener("click", (event) => {
  const link = event.target.closest("a.cite");
  if (!link) return;
  event.preventDefault();
  showCitation(link.dataset.path.replace(/^repo\//, ""), Number(link.dataset.start), Number(link.dataset.end));
});

// ---- Model picker: models installed in Ollama on this computer; cloud and embedding models are not offered.
async function loadModels() {
  try {
    const data = await api("/api/models");
    const select = $("model");
    const models = data.models.length ? data.models : [{name: data.current}];
    select.replaceChildren(...models.map((model) => {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = model.name + (model.parameters ? ` · ${model.parameters}` : "") + (model.name === data.default ? t(" · default") : "");
      return option;
    }));
    select.value = data.current;
    $("model-name").textContent = data.current;
    $("model-status").textContent = data.current === data.default
      ? t("The default, and the model CodeHaven is tested with.")
      : t("Not the tested default ({name}), so answers may differ. The first answer is slower while it loads.", {name: data.default});
  } catch (error) {
    $("model-status").textContent = error.message;
  }
}
// Choosing another model: the server checks it is installed before using it.
$("model").addEventListener("change", async () => {
  const name = $("model").value;
  busy = true; updateComposer();
  $("model-status").textContent = t("Checking {name}…", {name});
  try {
    await api("/api/model", {method: "POST", body: JSON.stringify({name})});
  } catch (error) {
    toast(error.message);
  } finally {
    busy = false; updateComposer(); loadModels();
  }
});

// ---- Chat.
function clearChat() {
  $("messages-inner").replaceChildren($("empty"));
  $("empty").hidden = false;
  transcript = []; $("save-chat").disabled = true;
}
// ---- Saving to a file: a download link that exists only for the click. Nothing is sent anywhere.
function saveFile(name, text, type) {
  const link = Object.assign(document.createElement("a"), {href: URL.createObjectURL(new Blob([text], {type})), download: name});
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
// The chat on the page as questions and answers, kept so it can be saved as a file.
let transcript = [];
// Add one question and its answer to the transcript, and allow saving once there is something to save.
function keepTurn(question, answer, issues, stopped) {
  transcript.push({question, answer, issues: issues || [], stopped: Boolean(stopped)});
  $("save-chat").disabled = false;
}
// The chat as Markdown: a heading per question, the answer under it, and any warning as a quotation.
function chatMarkdown() {
  const about = $("repo-label").textContent.trim();
  const parts = [`# CodeHaven: ${about}`, `${new Date().toISOString().slice(0, 16).replace("T", " ")} UTC · ${$("model").value}`];
  for (const turn of transcript) {
    parts.push(`## ${turn.question.replace(/\s+/g, " ")}`, turn.answer || "—");
    if (turn.stopped) parts.push(`> ${t("Stopped before it finished.")}`);
    if (turn.issues.length) parts.push(`> **${t("Unverified")}**\n` + turn.issues.map((issue) => `> - ${issue}`).join("\n"));
  }
  return parts.join("\n\n") + "\n";
}
$("save-chat").addEventListener("click", () => {
  if (!transcript.length) return;
  const name = ($("repo-label").textContent.trim().replace(/[^\wء-ي.-]+/g, "-") || "chat") + "-chat.md";
  saveFile(name, chatMarkdown(), "text/markdown");
});
// Add an empty message box of one kind (user, assistant, or note) and return it to be filled.
function addMessage(kind) {
  $("empty").hidden = true;
  const element = document.createElement("div");
  element.className = "msg " + kind;
  if (kind === "user") element.dir = "auto";  // An Arabic question is shown right to left.
  $("messages-inner").append(element);
  return element;
}
// Keep the newest message in view.
function scrollDown() { const box = $("messages"); box.scrollTop = box.scrollHeight; }
// Enable or disable every control to match the state: nothing open, busy, or ready.
function updateComposer() {
  const ready = indexed && !busy;
  $("question").disabled = !ready;
  // While an answer or flashcards are being written, the button that started them stops them.
  const stopLabel = t(stopping ? "Stopping…" : "Stop");
  $("send").textContent = stoppable ? stopLabel : t("Ask");
  $("send").classList.toggle("primary", !stoppable); $("send").classList.toggle("stop", stoppable);
  $("send").disabled = stoppable ? stopping : !ready || !$("question").value.trim();
  $("question").placeholder = t(!indexed ? "Open a project or upload a file first" : busy ? "CodeHaven is working…"
    : uploaded ? "Ask about this file" : "Ask about this repository");
  $("language").disabled = busy;
  $("new-chat").disabled = !indexed || busy;
  $("model").disabled = busy;
  // While a repository is indexed or downloaded, the button that started it stops it.
  $("index").textContent = indexing === "index" ? stopLabel : t("Index");
  $("index").disabled = indexing === "index" ? stopping : busy;
  $("upload").disabled = $("attach").disabled = busy;
  $("clone").textContent = indexing === "clone" ? stopLabel : t("Download and index");
  $("clone").disabled = indexing === "clone" ? stopping : busy;
  $("forget").disabled = busy || uploaded || !(savedCount || chatCount);
  $("tab-map").disabled = $("tab-cards").disabled = !indexed;
  $("map-draw").disabled = !indexed;
  $("cards-make").textContent = stoppable ? stopLabel : t("Make flashcards");
  $("cards-make").classList.toggle("primary", !stoppable); $("cards-make").classList.toggle("stop", stoppable);
  $("cards-make").disabled = stoppable ? stopping : !ready;
  for (const button of document.querySelectorAll("[data-choose]")) button.hidden = uploaded;
  showChosen();
  for (const button of $("suggestions").querySelectorAll("button")) button.disabled = !ready;
}
// Shorten a step line from the server's log for the status shown while waiting.
function statusText(line) {
  // The server's step lines are in English; the Arabic page shows only how far along the steps are.
  const step = LANG === "ar" && line.match(/^\s*Step (\d+) \(max (\d+)\)/);
  if (step) return t("Step {step} of at most {most}…", {step: step[1], most: step[2]});
  return line.replace(/^\s+/, "").replace(/\s*\(loading and prompt processing can take time\)\.\.\.$/, "…");
}

// Stop the answer or flashcards being written. The request that is streaming them ends with a "stopped" event.
async function stopNow() {
  if (!stoppable || stopping) return;
  stopping = true; updateComposer();
  try { await api("/api/stop", {method: "POST", body: "{}"}); }
  catch (error) { stopping = false; updateComposer(); toast(error.message); }
}
// Send one question and show the answer as it streams in; then its checks, the steps taken, and a Copy button.
async function ask(question) {
  question = question.trim();
  if (!question || busy || !indexed) return;
  busy = stoppable = true;
  $("question").value = ""; resize(); updateComposer();
  addMessage("user").textContent = question;
  const card = addMessage("assistant");
  const status = document.createElement("div"); status.className = "status";
  const dot = document.createElement("span"); dot.className = "dot";
  const statusLabel = document.createElement("span"); statusLabel.textContent = t("Starting…");
  status.append(dot, statusLabel);
  const answer = document.createElement("div"); answer.className = "answer";
  card.append(status, answer);
  scrollDown();
  const logs = [];
  let streamed = "", phase = t("Starting…");
  const started = Date.now();
  const timer = setInterval(() => { statusLabel.textContent = `${phase} ${t("{seconds}s", {seconds: Math.round((Date.now() - started) / 1000)})}`; }, 1000);
  const finish = () => { clearInterval(timer); busy = stoppable = stopping = false; updateComposer(); $("question").focus(); };
  try {
    let done = null;
    for await (const event of streamEvents("/api/ask", {question})) {
      if (event.type === "log") {
        logs.push(event.text);
        const step = event.text.trim().match(/^Step \d+ \(max \d+\): (.*)$/);
        if (step) phase = statusText(step[0]);
      } else if (event.type === "text") {
        streamed += event.text;
        answer.innerHTML = render(streamed);
        phase = t("Writing the answer…");
        scrollDown();
      } else if (event.type === "done" || event.type === "error" || event.type === "stopped") {
        done = event;
      }
    }
    status.remove();
    if (!done) throw new Error(t("The connection closed before the answer finished."));
    if (done.type === "error") throw new Error(done.message);
    if (done.type === "stopped") {
      // What was written so far stays on the page, marked as unfinished.
      const note = document.createElement("div"); note.className = "stopped";
      note.textContent = t("Stopped. This question is not remembered in follow-ups.");
      card.append(note);
      if (streamed) keepTurn(question, streamed, [], true);
      return;
    }
    answer.innerHTML = render(done.complete ? done.answer : (streamed || t("No final answer: the step limit was reached before CodeHaven finished.")));
    if (done.issues.length) {
      const checks = document.createElement("div"); checks.className = "checks";
      const title = document.createElement("strong"); title.textContent = t("Unverified: check the cited lines before relying on this answer.");
      const list = document.createElement("ul");
      for (const issue of done.issues) { const item = document.createElement("li"); item.dir = "auto"; item.textContent = issue; list.append(item); }
      checks.append(title, list); card.append(checks);
    }
    keepTurn(question, done.complete ? done.answer : streamed, done.issues);
    setSaved({saved: done.saved}, done.kept);
    setChats(done.chats);
    $("memory-status").textContent = t("Follow-ups remember {remembered} of the last {memory} questions. Start a new chat when you change topic.", done);
    const shownLines = new Set((done.answer || "").split("\n").map((line) => line.trim()).filter(Boolean));
    const steps = logs.filter((line) => line.trim() && !shownLines.has(line.trim()) && !line.startsWith("Answer check:"));
    const details = document.createElement("details"); details.className = "steps";
    const summary = document.createElement("summary");
    summary.textContent = t("{steps} · {seconds}s · show what CodeHaven did", {steps: count(done.steps, "step"), seconds: Math.round((Date.now() - started) / 1000)});
    const pre = document.createElement("pre"); pre.textContent = steps.join("\n").replace(/\n{3,}/g, "\n\n").trim();
    details.append(summary, pre);
    const copy = document.createElement("button"); copy.className = "copy"; copy.textContent = t("Copy");
    copy.title = t("Copy this answer as text");
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(done.answer || streamed); copy.textContent = t("Copied"); setTimeout(() => { copy.textContent = t("Copy"); }, 1500); }
      catch { toast(t("This browser did not allow copying. Select the text and copy it instead.")); }
    });
    const foot = document.createElement("div"); foot.className = "msg-foot";
    foot.append(details, copy); card.append(foot);
  } catch (error) {
    status.remove();
    const box = document.createElement("div"); box.className = "error-box"; box.dir = "auto"; box.textContent = error.message;
    card.append(box);
  } finally {
    finish(); scrollDown();
  }
}

// ---- Wiring: the question box, the sidebar buttons, the dialogs, and the Escape key.
// The question box grows with its text, up to its largest height.
function resize() { const box = $("question"); box.style.height = "auto"; box.style.height = Math.min(box.scrollHeight, 200) + "px"; }
$("question").addEventListener("input", () => { resize(); updateComposer(); });
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); ask($("question").value); }
});
$("send").addEventListener("click", () => stoppable ? stopNow() : ask($("question").value));
$("index").addEventListener("click", indexRepository);
$("path").addEventListener("keydown", (event) => { if (event.key === "Enter") indexRepository(); });
$("browse").addEventListener("click", openBrowser);
$("browser-cancel").addEventListener("click", closeOverlays);
$("browser-use").addEventListener("click", () => { $("path").value = browsing; closeOverlays(); indexRepository(); });
$("drawer-close").addEventListener("click", closeOverlays);
$("overlay").addEventListener("click", closeOverlays);
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeOverlays(); });
$("new-chat").addEventListener("click", async () => {
  try {
    await api("/api/reset", {method: "POST", body: "{}"});
    clearChat();
    $("memory-status").textContent = t("New chat started. Follow-up questions remember the last 8 questions.");
  } catch (error) { toast(error.message); }
});

// ---- Views: chat, and two study aids for one file or a group of files.
// Show one view and mark its tab as pressed.
function setView(name) {
  for (const view of ["chat", "map", "cards"]) {
    $("view-" + view).hidden = view !== name;
    $("tab-" + view).setAttribute("aria-pressed", String(view === name));
  }
}
for (const view of ["chat", "map", "cards"]) $("tab-" + view).addEventListener("click", () => setView(view));

// ---- The files a map or flashcards are about. An uploaded file is always the choice.
let selected = [], fileList = null, picking = new Set();
// Show the chosen files in the bar above the map and above the flashcards.
function showChosen() {
  const text = uploaded ? $("repo-label").textContent : !selected.length ? t("No files chosen")
    : selected.length === 1 ? selected[0] : `${count(selected.length, "file")}: ${selected.map((path) => path.split("/").pop()).join(", ")}`;
  for (const label of document.querySelectorAll("[data-chosen]")) {
    label.textContent = text;
    label.title = selected.join("\n");
    label.classList.toggle("none", !uploaded && !selected.length);
  }
}
// Forget the chosen files, the map, and the cards when something else is opened.
function resetStudy() {
  selected = []; fileList = null; mapData = null; cards = [];
  drawGraph([], []);
  $("map-back").hidden = true;
  $("deck").hidden = true;
  for (const id of ["map-status", "cards-status"]) { $(id).classList.remove("error"); $(id).textContent = STUDY_HINTS[id]; }
}
// List the files that match the filter, the first 300 at most, each with a checkbox.
function showPicker() {
  const filter = $("picker-filter").value.trim().toLowerCase();
  const shown = fileList.files.filter((path) => path.toLowerCase().includes(filter));
  const rows = shown.slice(0, 300).map((path) => {
    const row = document.createElement("label");
    const box = Object.assign(document.createElement("input"), {type: "checkbox", checked: picking.has(path)});
    box.addEventListener("change", () => { if (box.checked) picking.add(path); else picking.delete(path); showPickerNote(); });
    row.append(box, path);
    return row;
  });
  if (shown.length > rows.length) rows.push(Object.assign(document.createElement("div"), {className: "index-status", textContent: t("{count} more files: type in the filter to narrow the list.", {count: shown.length - rows.length})}));
  if (!shown.length) rows.push(Object.assign(document.createElement("div"), {className: "index-status", textContent: t("No files match.")}));
  $("picker-list").replaceChildren(...rows);
  showPickerNote();
}
// Say how many files are ticked in the picker, and how many each study aid takes.
function showPickerNote() {
  const limits = fileList.limits;
  $("picker-note").textContent = t("{count} chosen. A map takes up to {map} files, flashcards up to {flashcards}.", {count: picking.size, ...limits});
}
// Open the file picker. The file list is fetched once and kept until something else is opened.
async function openPicker() {
  $("overlay").classList.add("open"); $("picker").classList.add("open");
  picking = new Set(selected);
  try {
    if (!fileList) { $("picker-list").textContent = t("Loading…"); fileList = await api("/api/files"); }
    showPicker();
    $("picker-filter").focus();
  } catch (error) {
    $("picker-list").textContent = error.message;
  }
}
for (const button of document.querySelectorAll("[data-choose]")) button.addEventListener("click", openPicker);
$("picker-filter").addEventListener("input", () => { if (fileList) showPicker(); });
$("picker-clear").addEventListener("click", () => { picking.clear(); if (fileList) showPicker(); });
$("picker-done").addEventListener("click", () => { selected = [...picking]; closeOverlays(); showChosen(); });
const STUDY_HINTS = {"map-status": $("map-status").textContent, "cards-status": $("cards-status").textContent};
// Show an error in a study aid's status line.
function studyError(id, message) { $(id).classList.add("error"); $(id).textContent = message; }

// ---- Structure map: boxes and arrows drawn as SVG by this page; nothing is loaded from anywhere.
const SVG = $("map-svg").namespaceURI;  // Taken from the page, so no web address is written anywhere in it.
const BOX = {width: 214, height: 46, gapX: 70, gapY: 14, pad: 8};
let mapData = null;
// Make one SVG element with its attributes and, if given, its text.
function svg(tag, attributes = {}, text) {
  const element = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  if (text !== undefined) element.textContent = text;
  return element;
}
// A label that is too long keeps its end, which for a path is the file name.
function shorten(text, width = 27) { return text.length <= width ? text : "…" + text.slice(text.length - width + 1); }
// Columns from left to right: a box sits one column after the last box that points at it. Arrows that close a
// loop are drawn dashed and do not move anything, so two files that import each other still fit in a picture.
function layout(nodes, edges) {
  const out = new Map(nodes.map((node) => [node.id, []]));
  for (const edge of edges) out.get(edge.from).push(edge);
  const state = new Map(), back = new Set();
  const visit = (id) => {
    state.set(id, 1);
    for (const edge of out.get(id)) { if (state.get(edge.to) === 1) back.add(edge); else if (!state.get(edge.to)) visit(edge.to); }
    state.set(id, 2);
  };
  for (const node of nodes) if (!state.get(node.id)) visit(node.id);
  const column = new Map(nodes.map((node) => [node.id, 0]));
  for (let round = 0, moved = true; moved && round <= nodes.length; round++) {
    moved = false;
    for (const edge of edges) {
      if (!back.has(edge) && column.get(edge.to) < column.get(edge.from) + 1) { column.set(edge.to, column.get(edge.from) + 1); moved = true; }
    }
  }
  const linked = new Set(edges.flatMap((edge) => [edge.from, edge.to]));
  const columns = [];
  for (const node of nodes) if (linked.has(node.id)) (columns[column.get(node.id)] ||= []).push(node);
  // Order each column by where its callers sit in the column before, which keeps most arrows from crossing.
  const row = new Map();
  columns.forEach((list, index) => {
    if (index) {
      const weight = (node) => {
        const rows = edges.filter((edge) => edge.to === node.id && row.has(edge.from)).map((edge) => row.get(edge.from));
        return rows.length ? rows.reduce((a, b) => a + b, 0) / rows.length : 1e9;
      };
      list.sort((a, b) => weight(a) - weight(b));
    }
    list.forEach((node, position) => row.set(node.id, position));
  });
  const place = new Map(), step = BOX.height + BOX.gapY;
  const tallest = Math.max(0, ...columns.map((list) => list.length));
  columns.forEach((list, index) => list.forEach((node, position) => place.set(node.id, {
    x: BOX.pad + index * (BOX.width + BOX.gapX), y: BOX.pad + position * step + (tallest - list.length) * step / 2})));
  // Boxes with no arrows go in a grid underneath.
  const loose = nodes.filter((node) => !linked.has(node.id));
  const across = Math.max(3, columns.length), top = BOX.pad + (tallest ? tallest * step + 18 : 0);
  loose.forEach((node, index) => place.set(node.id, {x: BOX.pad + (index % across) * (BOX.width + 16), y: top + Math.floor(index / across) * step}));
  return {place, back};
}
// Draw the boxes and arrows, and return how many arrows there are. Repeated arrows, arrows from a box to itself,
// and arrows to a box that is not drawn are left out.
function drawGraph(nodes, edges) {
  const picture = $("map-svg");
  const known = new Set(nodes.map((node) => node.id)), seen = new Set();
  edges = edges.filter((edge) => {
    const key = edge.from + "\n" + edge.to;
    if (edge.from === edge.to || !known.has(edge.from) || !known.has(edge.to) || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  const {place, back} = layout(nodes, edges);
  const marker = svg("marker", {id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse"});
  marker.append(svg("path", {d: "M0 0 10 5 0 10Z", class: "arrow"}));
  const defs = svg("defs"); defs.append(marker);
  const lines = edges.map((edge) => {
    const from = place.get(edge.from), to = place.get(edge.to);
    const y1 = from.y + BOX.height / 2, y2 = to.y + BOX.height / 2, x1 = from.x + BOX.width;
    const forward = to.x > from.x;
    // An arrow that goes back, or stays in its column, leaves and arrives on the right-hand side.
    const d = forward ? `M${x1} ${y1} C${x1 + 36} ${y1} ${to.x - 36} ${y2} ${to.x} ${y2}`
      : `M${x1} ${y1} C${x1 + 44} ${y1} ${to.x + BOX.width + 44} ${y2} ${to.x + BOX.width} ${y2}`;
    const line = svg("path", {d, class: "edge" + (back.has(edge) || !forward ? " back" : ""), "marker-end": "url(#arrow)"});
    line.append(svg("title", {}, `${edge.from} → ${edge.to}${edge.title ? ": " + edge.title : ""}`));
    line.dataset.from = edge.from; line.dataset.to = edge.to;
    return line;
  });
  const boxes = nodes.map((node) => {
    const {x, y} = place.get(node.id);
    const group = svg("g", {class: `node ${node.kind || ""}${node.ghost ? " ghost" : ""}`, transform: `translate(${x} ${y})`});
    group.append(svg("title", {}, node.title || node.id), svg("rect", {width: BOX.width, height: BOX.height, rx: 8}),
                 svg("text", {x: 12, y: 19, class: "label"}, shorten(node.label)), svg("text", {x: 12, y: 36, class: "sub"}, node.sub));
    const near = new Set([node.id, ...edges.filter((edge) => edge.from === node.id || edge.to === node.id).flatMap((edge) => [edge.from, edge.to])]);
    const focus = (on) => {
      picture.classList.toggle("focus", on);
      for (const line of lines) line.classList.toggle("hot", on && (line.dataset.from === node.id || line.dataset.to === node.id));
      boxes.forEach((box, index) => box.classList.toggle("hot", on && near.has(nodes[index].id)));
    };
    group.addEventListener("mouseenter", () => focus(true));
    group.addEventListener("mouseleave", () => focus(false));
    if (node.open) {
      group.setAttribute("tabindex", "0"); group.setAttribute("role", "button");
      group.addEventListener("click", node.open);
      group.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); node.open(); } });
    }
    return group;
  });
  const places = [...place.values()];
  picture.setAttribute("width", places.length ? Math.max(...places.map((spot) => spot.x)) + BOX.width + 60 : 0);
  picture.setAttribute("height", places.length ? Math.max(...places.map((spot) => spot.y)) + BOX.height + BOX.pad : 0);
  picture.replaceChildren(defs, ...lines, ...boxes);
  return edges.length;
}
// An English count with its noun: "1 file", "3 files".
function plural(count, word) { return `${count} ${word}${count === 1 ? "" : "s"}`; }
// The map inside one file: its functions and classes, and which calls which.
function showFileMap(file) {
  const nodes = file.parts.map((part) => ({id: part.name, label: part.name, kind: part.kind, title: t("{name} · click to see the code", part),
    sub: `${part.kind === "part" ? "" : part.kind + " · "}${t("lines {start}-{end}", part)}`, open: () => showCitation(file.path, part.start, part.end)}));
  const calls = drawGraph(nodes, file.parts.flatMap((part) => part.calls.map((name) => ({from: part.name, to: name, title: t("calls")}))));
  $("map-back").hidden = mapData.files.length < 2;
  $("map-status").textContent = !nodes.length ? t("{path} has {lines} but no functions or classes CodeHaven can name, so there is nothing to draw.", {path: file.path, lines: count(file.lines, "line")})
    : t("{path}: {parts}, {calls} between them{limited}. An arrow points from the caller to what it calls. Click a box to see its code.",
        {path: file.path, parts: count(nodes.length, "part"), calls: count(calls, "call"), limited: file.parts_limited ? t(" (only the first parts are shown)") : ""});
}
// The map of several files: which imports which. Clicking a file shows the map inside it.
function showFilesMap() {
  const nodes = mapData.files.map((file) => ({id: file.path, label: file.path, kind: "file", title: t("{path} · click to see inside", file),
    sub: `${count(file.parts.length, "part")} · ${count(file.lines, "line")}`, open: () => showFileMap(file)}));
  nodes.push(...mapData.neighbours.map((path) => ({id: path, label: path, ghost: true, sub: t("not chosen"), title: t("{path} · linked to a chosen file, but not chosen", {path})})));
  const links = drawGraph(nodes, mapData.links.map((link) => ({from: link.from, to: link.to, title: link.names.length ? t("imports {names}", {names: link.names.join(", ")}) : t("imports it")})));
  $("map-back").hidden = true;
  $("map-status").textContent = t("{files}, {imports}{limited}. An arrow points from a file to the file it imports; dashed boxes are files you did not choose. Click a file to see inside it.",
    {files: count(mapData.files.length, "file"), imports: count(links, "import"), limited: mapData.limited ? t(" (limits reached, so some are left out)") : ""});
}
// Ask the server for the structure of the chosen files and draw it.
async function drawMap() {
  if (!uploaded && !selected.length) { openPicker(); return; }
  $("map-status").classList.remove("error");
  $("map-status").textContent = t("Reading the code…");
  try {
    mapData = await api("/api/map", {method: "POST", body: JSON.stringify({paths: selected})});
    if (mapData.files.length === 1 && !mapData.neighbours.length) showFileMap(mapData.files[0]); else showFilesMap();
  } catch (error) {
    drawGraph([], []);
    studyError("map-status", error.message);
  }
}
$("map-draw").addEventListener("click", drawMap);
$("map-back").addEventListener("click", showFilesMap);

// ---- Flashcards: the local model writes them one file at a time; the lines behind each answer are looked up, not guessed.
let cards = [], cardAt = 0, cardAnswer = false;
// Show the current card, on its question side or its answer side.
function showCard() {
  $("deck").hidden = !cards.length;
  if (!cards.length) return;
  const card = cards[cardAt];
  $("flashcard").classList.toggle("answer", cardAnswer);
  $("card-side").textContent = t(cardAnswer ? "Answer" : "Question");
  $("card-text").innerHTML = inline(cardAnswer ? card.answer : card.question);
  $("card-text").dir = direction(cardAnswer ? card.answer : card.question);
  const where = card.start ? `${card.path}:${card.start}${card.end !== card.start ? "-" + card.end : ""}` : "";
  $("card-source").innerHTML = !cardAnswer ? esc(card.path) : where ? inline(t("See {where}", {where})) + (card.symbol ? ` (<code>${esc(card.symbol)}</code>)` : "")
    : esc(t("{path}: the part this answer names was not found, so check it against the file.", card));
  $("card-count").textContent = t("Card {at} of {total}", {at: cardAt + 1, total: cards.length});
  $("card-flip").textContent = t(cardAnswer ? "Show question" : "Show answer");
  $("card-prev").disabled = cardAt === 0;
  $("card-next").disabled = cardAt === cards.length - 1;
}
// Go to the next or previous flashcard, showing its question side.
function moveCard(step) { cardAt = Math.min(cards.length - 1, Math.max(0, cardAt + step)); cardAnswer = false; showCard(); }
// Turn the current flashcard over.
function flipCard() { cardAnswer = !cardAnswer; showCard(); }
// Ask the model for cards. They arrive one file at a time and are shown as they come.
async function makeCards() {
  if (busy || !indexed) return;
  if (!uploaded && !selected.length) { openPicker(); return; }
  busy = stoppable = true; updateComposer();
  const status = $("cards-status"), failed = [], started = Date.now();
  status.classList.remove("error");
  cards = []; cardAt = 0; cardAnswer = false; showCard();
  let phase = t("Starting…"), finished = false, stopped = false;
  const tick = () => { status.textContent = `${phase} ${t("{seconds}s", {seconds: Math.round((Date.now() - started) / 1000)})}`; };
  const timer = setInterval(tick, 1000);
  tick();
  try {
    for await (const event of streamEvents("/api/flashcards", {paths: selected, count: Number($("cards-count").value), language: LANG})) {
      if (event.type === "file") { phase = t("Writing cards for {path} (file {index} of {of})…", event); tick(); }
      else if (event.type === "cards") { cards.push(...event.cards); showCard(); }
      else if (event.type === "error") failed.push(`${event.path}: ${event.message}`);
      else if (event.type === "done") finished = true;
      else if (event.type === "stopped") stopped = true;
    }
    clearInterval(timer);
    if (stopped) { status.textContent = t("Stopped. ") + t("{cards} kept.", {cards: count(cards.length, "card")}); return; }
    if (!finished) throw new Error(t("The connection closed before the cards were finished."));
    status.classList.toggle("error", !cards.length);
    status.textContent = (cards.length ? t("{cards} in {seconds}s. Click the card, or press Space, to turn it over; the arrow keys move between cards. ",
                                           {cards: count(cards.length, "card"), seconds: Math.round((Date.now() - started) / 1000)})
                                       : t("No cards were made. "))
      + (failed.length ? t("Skipped {files}", {files: failed.join("; ")}) : "");
  } catch (error) {
    clearInterval(timer);
    studyError("cards-status", error.message);
  } finally {
    busy = stoppable = stopping = false; updateComposer();
  }
}
$("cards-make").addEventListener("click", () => stoppable ? stopNow() : makeCards());
$("flashcard").addEventListener("click", (event) => { if (!event.target.closest("a")) flipCard(); });
$("card-flip").addEventListener("click", flipCard);
$("card-prev").addEventListener("click", () => moveCard(-1));
$("card-next").addEventListener("click", () => moveCard(1));
$("cards-shuffle").addEventListener("click", () => {
  for (let i = cards.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [cards[i], cards[j]] = [cards[j], cards[i]]; }
  cardAt = 0; cardAnswer = false; showCard();
});
$("cards-copy").addEventListener("click", async () => {
  const text = cards.map((card) => `Q: ${card.question}\nA: ${card.answer}\n${card.start ? `${card.path}:${card.start}-${card.end}` : card.path}`).join("\n\n");
  try { await navigator.clipboard.writeText(text); $("cards-copy").textContent = t("Copied"); }
  catch { $("cards-copy").textContent = t("Copying is blocked by the browser"); }
  setTimeout(() => { $("cards-copy").textContent = t("Copy all as text"); }, 2000);
});
// One card per line, question then answer, separated by a tab: the plain-text form Anki imports.
$("cards-save").addEventListener("click", () => {
  const cell = (text) => text.replace(/[\t\r\n]+/g, " ").trim();
  const rows = cards.map((card) => `${cell(card.question)}\t${cell(card.answer)} (${card.start ? `${card.path}:${card.start}-${card.end}` : card.path})`);
  saveFile("codehaven-flashcards.txt", rows.join("\n") + "\n", "text/tab-separated-values");
});
// Keys in the flashcards view: Space or Enter turns the card over, the arrow keys move between cards.
document.addEventListener("keydown", (event) => {
  if ($("view-cards").hidden || !cards.length || $("overlay").classList.contains("open") || event.target.closest("input, select, textarea, button")) return;
  if (event.key === " " || event.key === "Enter") { event.preventDefault(); flipCard(); }
  else if (event.key === "ArrowRight") moveCard(LANG === "ar" ? -1 : 1);  // In the Arabic page, next is on the left.
  else if (event.key === "ArrowLeft") moveCard(LANG === "ar" ? 1 : -1);
});
// The choices for cards per file.
$("cards-count").replaceChildren(...[3, 4, 5, 6].map((count) => Object.assign(document.createElement("option"), {value: count, textContent: count, selected: count === 5})));

// ---- Start-up: show what the server already has open.
showRecent();
loadModels();
// A reloaded page shows no earlier messages, so the server forgets them too: follow-ups never use hidden context.
api("/api/status").then((summary) => {
  setIndexed(summary);
  if (summary.repository) api("/api/reset", {method: "POST", body: "{}"}).catch(() => {});
}).catch((error) => { $("index-status").textContent = error.message; });
updateComposer();
