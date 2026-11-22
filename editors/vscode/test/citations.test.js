// Tests for the parts of the extension that do not need VS Code. Run with: npm test
"use strict";
const test = require("node:test");
const assert = require("node:assert");
const {findCitations, parseCitation, shellQuote, pageLink} = require("../citations");

test("citations are found with their path and line range", () => {
  const line = "- inventory/store.py:27-29 `Store.low_stock`: uses < (see repo/config.py:2).";
  assert.deepStrictEqual(findCitations(line).map(({path, start, end}) => [path, start, end]),
    [["inventory/store.py", 27, 29], ["config.py", 2, 2]]);
  const [first] = findCitations(line);
  assert.strictEqual(line.slice(first.index, first.index + first.length), "inventory/store.py:27-29");
});

test("times, addresses, and backwards ranges are not citations", () => {
  for (const line of ["It took 10:30 minutes", "http://127.0.0.1:8737/?token=abc", "store.py:9-3", "version 2.5:14", "see store.py"]) {
    assert.deepStrictEqual(findCitations(line), [], line);
  }
});

test("a typed citation is parsed, with or without the repo/ label", () => {
  assert.deepStrictEqual(parseCitation("  repo/src/app.ts:12 "), {index: 0, length: 18, path: "src/app.ts", start: 12, end: 12});
  assert.strictEqual(parseCitation("nothing here"), null);
});

test("questions are quoted as one shell argument", () => {
  assert.strictEqual(shellQuote("What's `add`?", false), `'What'\\''s \`add\`?'`);
  assert.strictEqual(shellQuote("What's `add`?", true), "'What''s `add`?'");
});

test("the page link is read from what CodeHaven prints", () => {
  assert.strictEqual(pageLink("CodeHaven is running at:\n  http://127.0.0.1:8737/?token=aB-c_1\nOnly this machine"), "http://127.0.0.1:8737/?token=aB-c_1");
  assert.strictEqual(pageLink("Error: Local Ollama request failed"), null);
});
