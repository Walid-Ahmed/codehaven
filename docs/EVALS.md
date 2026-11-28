# Eval results

This file is the record of how the local model scores on CodeHaven's fixed question sets, so that a change can be compared with what came before it. Add a row when a change is measured; do not rewrite old rows.

## The question sets

| Set | Repository | Questions | What it is for |
|---|---|---|---|
| `evals/tasks.json` | [evals/sample_repo](../evals/sample_repo), 12 files | 12 | The set prompts were first tuned against |
| `evals/bookings_tasks.json` | [evals/bookings_repo](../evals/bookings_repo), 6 files | 20 | A second small repository, to catch tuning that fits only the first |
| `--suite requests` | [requests](https://github.com/psf/requests) v2.32.3, 102 files | 12 | A real Python library |
| `--suite click` | [click](https://github.com/pallets/click) 8.1.8, 132 files | 11 | A real Python library with a large public surface |
| `--suite express` | [express](https://github.com/expressjs/express) 4.21.2, 223 files | 11 | A real JavaScript library, written in the CommonJS style |
| `--suite flask` | [flask](https://github.com/pallets/flask) 3.1.0, 228 files | 12 | Held out: a Python library the v0.11 changes never saw |
| `--suite cobra` | [cobra](https://github.com/spf13/cobra) v1.8.1, 55 files | 11 | Held out: a Go library |
| `--suite axios` | [axios](https://github.com/axios/axios) v1.7.9, 210 files | 11 | Held out: a JavaScript library written as ES modules |

The three real repositories are not kept in this repository. Download them once, while online, and the evals then run offline like the others:

```sh
python3 scripts/fetch_eval_repos.py
python3 -m offline_agent.evaluation --suite requests
```

The script is a developer tool and the only step that uses the internet; the installed package never runs it. Each repository is pinned to one release commit in [evals/real/repos.json](../evals/real/repos.json), and a download that is not exactly that commit is refused, so the questions always meet the code they were written for.

Every question is checked by regular expressions on the answer, written from the source at the pinned commit. A pass means the answer contains the facts; **verified** means it also passed CodeHaven's own answer checks, so every citation points at lines the model was shown. Each real-repository question was written before the change it later measured, except where a row says otherwise.

## Results

All runs: `qwen2.5-coder:7b`, default settings, an Apple M4 with 24 GB, no other Ollama client running.

### Real repositories

| Date | Code | requests | click | express | Total | Verified |
|---|---|---|---|---|---|---|
| 2026-11-22 | Baseline, before any change measured here | 10/12 | 8/11 | 6/11 | 24/34 (71%) | 16/34 |
| 2026-11-22 | + the definitions a question names, and the constants they use | 11/12 | 9/11 | 8/11 | 28/34 (82%) | 17/34 |
| 2026-11-22 | + a brief-answer line, citing a shown line, inheriting classes | 12/12 | 9/11 | 8/11 | 29/34 (85%) | 15/34 |
| 2026-11-22 | + answer text of a plain question capped at 800 characters (v0.11.0) | 12/12 | 10/11 | 9/11 | 31/34 (91%) | 21/34 |
| 2026-11-22 | v0.11.0 run again, as the baseline for v0.12 | 12/12 | 10/11 | 9/11 | 31/34 (91%) | 21/34 |
| 2026-11-22 | + quoted settings and dotted names searched, evidence merged instead of dropped, uncited names cited, exported JavaScript members found | 12/12 | 10/11 | 10/11 | 32/34 (94%) | 31/34 |
| 2026-11-22 | + project files up to 1 MB, and names searched within a named file (v0.12.0) | 12/12 | 10/11 | 10/11 | 32/34 (94%) | 31/34 |
| 2026-11-22 | + values an answer reports looked for in the code (v0.13.0) | 12/12 | 10/11 | 10/11 | 32/34 (94%) | 31/34 |

Notes on the baseline:

- The first run scored requests at 9/12. One check wanted the literal `500` or `600` and failed the correct answer "status codes 400-599"; the check was loosened to accept `599` and `5xx`, and the saved answer was scored again. No other check was changed.
- The model finished at its first step on 32 of 34 questions, answering from the passages retrieval gave it. When those passages did not hold the answer, it did not search; it guessed. Examples: a default pool size of 100 (it is 10), `python-requests/2.25.1` as the User-Agent, and "does not read any environment variables" for a function that reads two.
- Five answers ran on until CodeHaven's loop guard stopped them. Four of those five were wrong.

What the later rows show:

- **Showing the definition did the most.** The model answers from what is in front of it. Putting the named function or class there, with the values of the constants it uses, fixed the guessed pool size, the missed environment variables, and the wrong default settings.
- **The brief-answer line alone did not stop the run-on answers.** In that row 10 answers were still cut by the loop guard after a correct first sentence, and a cut answer loses its citations, which is why fewer answers were verified than before.
- **The cap did.** With the answer text limited to 800 characters, no answer was cut by the loop guard, the citations after the text were written, and the whole run took 1,188 seconds where the baseline took 1,809.
- **Still wrong at the end:** the ANSI code for red (the question names no function, and retrieval brings the examples folder), the package behind `express.static` (the answer is one line, `exports.static = require('serve-static')`, that neither retrieval nor a named definition reaches), and a follow-up about the default of a setting.

The three changes after the baseline were made after reading the baseline's failures, on the same 34 questions, so the last row is partly fitted to them. The held-out repositories below are the fair test.

What the v0.12 rows show:

- **The rerun of v0.11.0 matched its recorded result exactly** (31/34, 21 verified), so the rows after it are compared with a fresh baseline, not with an older run.
- **Verified answers went from 21 to 31.** Of the 10 correct answers that v0.11.0 marked unverified, 6 had a citation range that no single retained evidence row held, 4 named a part that was "not found" (Express defines its helpers as `exports.compileETag = ...`, which the lookup refused, and two of the four were Express), and 2 had no supported citation at all; two answers had two of these problems. v0.12 checks a citation against every line shown, merges evidence into line ranges instead of dropping it when space runs low, finds members defined by assignment, and reads lines back from the file for a citation that names the right thing in the wrong place. That last step never worked before: retained evidence keeps only a 64-character snippet, and the rescue looked for the full text, which was never there.
- **`express.static` is now answered**, from the line `exports.static = require('serve-static')` that the dotted-name search finds. The ANSI code for red and the follow-up about the subdomain offset still fail.
- **Seeing files up to 1 MB changed nothing here.** Only one of these three repositories has a source file over the old 64 KB limit. It mattered on FastAPI, below.

### Held-out repositories

The fair test of the v0.11 answer changes. The 34 questions on flask, cobra, and axios were written from the source and committed before any run, on repositories none of the changes had been tried on. Each was then scored once on the code from before the answer changes and once on v0.11.0, with nothing adjusted afterwards.

| Date | Code | flask | cobra | axios | Total | Verified | Time |
|---|---|---|---|---|---|---|---|
| 2026-11-22 | Before the answer-quality changes | 10/12 | 8/11 | 10/11 | 28/34 (82%) | 19/34 | 2,232 s |
| 2026-11-22 | v0.11.0 | 12/12 | 8/11 | 10/11 | 30/34 (88%) | 20/34 | 1,639 s |

- **The gain holds, and it is smaller here.** Two more answers right out of 34, against seven more on the questions the changes were worked out on. The honest size of the improvement is nearer this one. Two questions out of 34 is also within what one run can differ from the next.
- **The old code did better on these repositories than on the first three** (82% against 71%), so there was less to gain.
- **About a quarter less time.** One old-code answer took all 12 steps and 374 seconds without reaching the answer; the same question was answered in one step on v0.11.0.
- **Verified answers barely moved** (19 to 20), unlike on the first three repositories. Short answers kept their citations, but on cobra most answers still cite lines the model was not shown.
- **Cobra did not move at all.** The same three questions fail before and after: the name of the hidden completion command, the variable that controls command sorting, and an annotation key. Each answer is one constant or variable that the question does not name, so no definition is shown. This is the same gap as the three questions still failing on the first repositories, and it is first on the plan's list.
- The fourth miss on v0.11.0 is on axios: it gave the XSRF names but not the default timeout of 0.

v0.12.0 on the same 34 held-out questions. The first row is the code before the 1 MB change, the second is v0.12.0:

| Date | Code | flask | cobra | axios | Total | Verified | Time |
|---|---|---|---|---|---|---|---|
| 2026-11-22 | v0.12, before seeing files over 64 KB | 12/12 | 8/11 | 10/11 | 30/34 (88%) | 30/34 | 1,557 s |
| 2026-11-22 | v0.12.0 | 12/12 | 7/11 | 10/11 | 29/34 (85%) | 28/34 | 1,370 s |

- **Verified answers went from 20 to 28 or 30**, the same gain as on the first repositories, on questions none of the v0.12 changes were worked out on.
- **Correct answers did not change in a way one run can show.** The one difference is `cobra-checkerr`: both runs gave the same explanation, and the second left out the words "to stderr", which the check looks for. Making cobra's two large test files visible changed what retrieval brought, and with it the wording.
- **The three cobra questions that fail every time still fail**, now with citations of real lines: the model cites the variable it found (`defaultCommandSorting`) instead of the one the question means (`EnableCommandSorting`).

### A large repository

FastAPI 0.115.6 has 2,338 visible files, ten times as many as the other repositories, and most of them are documentation in many languages, documentation examples, and tests; the library itself is 42 files. Its 12 questions were written from the source before any run.

| Date | Code | Correct | Verified | Time |
|---|---|---|---|---|
| 2026-11-22 | v0.11.0 | 9/12 | 7/12 | 587 s |
| 2026-11-22 | v0.12, before seeing files over 64 KB | 9/12 | 8/12 | 531 s |
| 2026-11-22 | v0.12.0 | 9/12 | 12/12 | 594 s |
| 2026-11-22 | v0.13.0 | 9/12 | 11/12 | 1,219 s |

The v0.13.0 run took about twice as long on every suite with the same tokens generated, and 99% of it was time spent waiting for Ollama, so the machine was slower that day; the check itself adds little.

- **Its central modules were invisible.** `routing.py` and `applications.py` are about 176 KB each, because of their long docstrings, and v0.11.0 ignored every file over 64 KB: it could not search, index, or read them. Every real repository here has such files: one each in requests and click, two in cobra, three in axios, and five in FastAPI. v0.12.0 sees project files of up to 1 MB. Only bounded parts of a file ever reach a prompt, so the prompt is no larger.
- **With them visible, every correct answer was also verified**, against 5 of 9 on v0.11.0.
- **The same three questions fail on every version.** The default OpenAPI version: the model reads `openapi: str = '3.0.0'` from the OpenAPI model class instead of the `"3.1.0"` default in `applications.py`. The callers of `solve_dependencies`: `routing.py` is now shown, with the matching lines and the functions they are in, and the model still names `request_response`. And the default version string, which the model says is not set. One v0.11.0 run passed that last one by quoting an example that happened to contain `0.1.0`.

### What the checks now say

v0.12 makes "verified" mean nearly every correct answer, and it costs something: wrong answers now pass the checks more often, because a wrong answer that cites real lines it misread passes them. On the first three repositories and FastAPI:

| Code | Correct answers that pass the checks | Wrong answers marked unverified |
|---|---|---|
| v0.11.0 | 26 of 40 (65%) | 4 of 6 |
| v0.12.0 | 39 of 41 (95%) | 1 of 5 |
| v0.13.0 | 39 of 41 (95%) | 2 of 5 |

The checks confirm that an answer cites lines the model was shown. They never confirmed that it read them right, and the wrong answers they used to catch were caught mostly because a wrong answer often had no citation at all. With a third of correct answers marked unverified, the mark was ignored; now it is rare, and it means a citation does not hold. Read "verified" as "you can check this quickly at the cited lines", not as "this is right".

### Checking the values an answer reports

v0.13 looks for each value an answer reports, a quoted string or a number of two or more digits, in the code the model was shown. A value on or within 3 lines of a cited line needs nothing. A value shown on another line gets a citation of that line, under a heading that says CodeHaven added it, and is not flagged. A value on no shown line was not read from the code, and the answer is marked unverified.

A prototype that instead required every value to be on the cited lines was scored first, on 72 saved answers: it flagged 5 of 8 wrong answers, and 14 of 64 correct ones, because correct answers often cite the definition and quote a value from a few lines away. That is why a value found elsewhere is cited rather than flagged.

All seven real repositories and the two small sets were run again on v0.13.0. The model's answers were the same as on v0.12.0, to the token, so every difference below comes from the check:

| | Correct answers (97) | Wrong answers (13) |
|---|---|---|
| Flagged for a value on no shown line | 1 | 3 |
| Of those, already unverified on v0.12.0 | 0 | 1 |
| Given a citation of the line a value is on | 10 | 5 |

- **Two wrong answers that passed every check are now caught.** FastAPI's default OpenAPI version, reported as `3.0.2` (it is `3.1.0`), and a Cobra flag annotation, reported by a name that is not in the code. The third, an ANSI escape code for red that the model made up, was already unverified.
- **One correct answer is flagged.** For Flask's default port, the model quoted a code block it wrote itself, `port: int = 5000`; the real default is set in a `click.option` above the lines it was shown. The value is right, but it was not read from the code, and the flag says exactly that.
- **One wrong answer now passes.** Asked what Cobra's `CheckErr` does, the answer gives the `Error:` prefix and exit code 1 but not that it prints to standard error. It cited no line in a checkable form, and the citation CodeHaven added for `Error:` made it checkable. The check confirms a value's source, not that an answer is complete.
- **The first full run flagged 7 correct answers, and three causes were fixed before the release.** "Lines 348-351" was read as the value 348; 422 was not found in `HTTP_422_UNPROCESSABLE_ENTITY`; and "a 10% discount" was not found in code that writes `0.9`. Line ranges and percentages are no longer treated as values, and a number is found wherever digits bound it. The 11 answers the first run flagged were run again with the fix; the table counts those results.

On the first three repositories and FastAPI the trade-off table above now reads 39 of 41 and 2 of 5. On the held-out three, correct answers that pass the checks went from 25 of 29 to 24 (the Flask port), and wrong answers marked unverified stayed at 2 of 5, one caught by a value and one, `CheckErr`, no longer caught. The small sets did not change: their values are on the cited lines.

### Fitting the reference excerpts to the prompt (v0.14)

Before v0.14, CodeHaven retrieved reference excerpts for every question, then preloaded the files and definitions the question names. When those filled the 12 KB prompt, compaction replaced the excerpts, the oldest message, with a one-line stub, so they were found for nothing and the model was told to fetch them again. That happened on 22 of the 80 real-repository questions. v0.14 retrieves again with the room that is left, or leaves the excerpts out when not even 2 KB is left (2 of the 80).

The 22 affected questions, on `qwen2.5-coder:7b`, one run each, before and after, on 2026-11-27 and 2026-11-28:

| Code | Correct | Correct and verified | Time for the 22 |
|---|---|---|---|
| v0.13 (excerpts compacted away) | 19/22 | 18/22 | 2,524 s |
| v0.14 (excerpts fitted) | 20/22 | 19/22 | 1,601 s |

- **One more answer right.** "Which functions in routing.py call solve_dependencies?" on FastAPI is now answered right and verified.
- **No answer lost.** The two FastAPI questions still wrong (the default OpenAPI version, and the default title and version) were wrong before too. One Flask answer changed from verified to unverified and one the other way round, so the verified count moved by one.
- **Faster, with a caveat.** The prompt is smaller, so Ollama processes less of it, but the first run shared the machine with other work, so read the time difference as a trend, not a measurement.

### Other models

The same 34 real-repository questions, the same code (v0.11.0), one run each on 2026-11-22. Per suite, in the order requests, click, express: the 14B model 12, 10, 9; llama3 10, 7, 8; the 0.5B model 5, 2, 6.

| Model | Size on disk | Correct | Verified | Time for 34 questions | Per question |
|---|---|---|---|---|---|
| `qwen2.5-coder:7b` (the default) | 4.7 GB | 31/34 (91%) | 21/34 | 1,188 s | 35 s |
| `qwen2.5:14b` | 9.0 GB | 31/34 (91%) | 28/34 | 2,031 s | 60 s |
| `llama3` (8B) | 4.7 GB | 25/34 (74%) | 25/34 | 909 s | 27 s |
| `qwen2.5:0.5b` | 0.4 GB | 13/34 (38%) | 9/34 | 479 s | 14 s |

- **The default is the right default.** The 7B coder model gets as many answers right as the model twice its size, in a little over half the time.
- **The 14B model is the careful one.** It is no more often right, but 28 of its answers passed every citation check against 21: it cites the lines it was shown instead of lines it remembers. Choose it when you have the memory (it needs about 10 GB loaded) and would rather wait than re-check a citation.
- **llama3 answers fastest of the usable three and is right least often**, mostly on click, where it missed four of eleven. Every answer it got right was also verified.
- **The 0.5B model is not usable for this.** It took three to four steps per question instead of one, and was right about a third of the time.

CodeHaven's prompts were written and tuned on `qwen2.5-coder:7b`, so the other models are measured on prompts made for a different model; a model could do better with prompts of its own. Loading time is included in the times, once per suite.

### The two small sets

These questions were not looked at while the answer-quality changes were made, so they are the fairer check that the changes help in general and break nothing.

| Date | Code | `tasks.json` | `bookings_tasks.json` | Verified |
|---|---|---|---|---|
| 2026-11-22 | Before the answer-quality changes | 12/14 | 12/16 | 12/14 and 13/16 |
| 2026-11-22 | v0.11.0, with all of them | 12/14 | 14/16 | 12/14 and 15/16 |
| 2026-11-22 | v0.12.0 | 13/14 | 14/16 | 12/14 and 16/16 |
| 2026-11-22 | v0.13.0 | 13/14 | 14/16 | 12/14 and 16/16 |

The same two sample questions fail before and after: a request to fix `Store.low_stock`, where the answer explains the fix without writing `<=`, and one about a slug. Each row is a single run, and results move by a question or so between runs, so read the bookings gain as "no worse", not as proof of two more.

On v0.12.0 the `Store.low_stock` question passes: the answer now names the `<=` it should become. The two bookings questions that fail ask for the date of a row in a 454-line table, "Founders Day" and "Regional Day 200". The preview of a long file is centred on the code names a question mentions, and a row's plain name is not one, so the row is never shown and the model says the holiday is not listed or guesses a date. v0.11.0 fails both in the same way when run again on them alone, so this is not new in v0.12, and it is on the plan.

These are short questions with one right answer each, on three libraries. They do not measure explanations of a whole project, large monorepos, or questions in Arabic.
