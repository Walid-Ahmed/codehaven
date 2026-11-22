// Finding CodeHaven's citations in a line of text. Kept apart from extension.js so it can be tested without VS Code.
"use strict";

// A citation is a file path, a colon, a line, and perhaps an end line: inventory/store.py:27-29.
// A leading repo/ is CodeHaven's label for the project and is not part of the path. The path must contain a
// file extension, so a clock time (10:30) or a host and port (127.0.0.1:8737) is not taken for one.
const CITATION = /(?<![\w./-])((?:repo\/)?(?:[\w.@+-]+\/)*[\w.@+-]*[A-Za-z_][\w@+-]*\.[A-Za-z]\w{0,7}):(\d{1,7})(?:-(\d{1,7}))?(?![\w/])/g;

// Every citation in one line: where it is in the line, and the path and line range it names.
function findCitations(text) {
  const found = [];
  for (const match of text.matchAll(CITATION)) {
    const start = Number(match[2]);
    const end = match[3] ? Number(match[3]) : start;
    if (start < 1 || end < start) continue;
    found.push({index: match.index, length: match[0].length, path: match[1].replace(/^repo\//, ""), start, end});
  }
  return found;
}

// One citation typed by hand, or null: used by the "Go to a cited line" command.
function parseCitation(text) {
  const found = findCitations(text.trim());
  return found.length ? found[0] : null;
}

// A value quoted for a POSIX shell or for PowerShell, so a question with spaces or quotes stays one argument.
function shellQuote(value, windows) {
  return windows ? `'${String(value).replace(/'/g, "''")}'` : `'${String(value).replace(/'/g, `'\\''`)}'`;
}

// The link CodeHaven prints when its page is ready, or null. The token in it is what lets the page talk to the server.
function pageLink(output) {
  const match = output.match(/http:\/\/127\.0\.0\.1:\d+\/\?token=[\w-]+/);
  return match ? match[0] : null;
}

module.exports = {findCitations, parseCitation, shellQuote, pageLink};
