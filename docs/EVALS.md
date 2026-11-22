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

The same two sample questions fail before and after: a request to fix `Store.low_stock`, where the answer explains the fix without writing `<=`, and one about a slug. Each row is a single run, and results move by a question or so between runs, so read the bookings gain as "no worse", not as proof of two more.

These are short questions with one right answer each, on three libraries. They do not measure explanations of a whole project, large monorepos, or questions in Arabic.
