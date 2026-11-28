# CodeHaven for VS Code

Ask [CodeHaven](https://github.com/Walid-Ahmed/codehaven) about the project you have open, and jump from a cited `path:line` to the code. The extension holds no model and sends nothing anywhere: it runs the `codehaven` command installed on this computer. Like CodeHaven, it never changes a file.

## What it adds

| Command | What it does |
|---|---|
| **CodeHaven: Open the Chat Panel** | A chat beside your code: answers stream in, a citation opens the cited lines, and with `--allow-edits` in `codehaven.extraArguments` a proposed change shows its diff with Apply and Decline |
| **CodeHaven: Explain the Selection in the Chat Panel** | Asks about the selected lines in the chat panel (also in the editor's right-click menu) |
| **CodeHaven: Ask a Question** | Runs your question on the open folder in a terminal named CodeHaven |
| **CodeHaven: Explain This File** | Asks CodeHaven to explain the file you are looking at |
| **CodeHaven: Explain the Selected Code** | Asks about the selected lines (also in the editor's right-click menu) |
| **CodeHaven: Trace the Function Under the Cursor** | Lists what the function calls and who calls it |
| **CodeHaven: Open the Chat Page for This Project** | Starts the local chat page and opens it in your browser |
| **CodeHaven: Stop the Chat Page** | Stops that page's server; closing the window does the same |
| **CodeHaven: Go to a Cited Line** | Opens a citation you type, such as `inventory/store.py:27-29` |

Every citation in terminal output, from this extension or from running `codehaven` yourself, becomes a link. Click it to open the file with the cited lines selected.

## Install

You need CodeHaven itself first; see the main README. Then, from this folder:

```sh
npx @vscode/vsce package
code --install-extension codehaven-0.2.0.vsix
```

`npx` downloads the packaging tool once. Without it, copy this folder to `~/.vscode/extensions/codehaven` and restart VS Code.

## Settings

- `codehaven.command`: the command to run. If CodeHaven is installed in a virtual environment, give the full path, such as `/path/to/.venv/bin/codehaven`.
- `codehaven.startOllama` (on by default): adds `--start-ollama`, so a local Ollama server is started when none is running.
- `codehaven.extraArguments`: more arguments for every run, such as `["--model", "qwen2.5:14b"]`.

## Tests

```sh
node --test test/citations.test.js test/extension.test.js
```

These run without VS Code, against a small stand-in for its API. To try the extension inside the VS Code installed on this computer, run:

```sh
test/integration/run.sh /path/to/.venv/bin/codehaven
```

It opens a VS Code window for a few seconds with a throwaway profile (your settings and extensions are not used or changed), and checks that the extension activates, registers its commands, opens a typed citation with the lines selected, sends a question to the CodeHaven terminal, starts the chat page from the real `codehaven` command on its 127.0.0.1 link, and answers in the chat panel. Nothing is downloaded. It also asks `/help` in the chat panel, which the local server answers without a model. Passed, 6 of 6, on VS Code 1.145.2 on macOS on 2026-11-28.
