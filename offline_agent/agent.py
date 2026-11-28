"""Local coding workflow orchestrated by a LangGraph StateGraph."""
import json
import re
import threading
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from . import arabic, changes, coder, guides, insights, messages as commit_messages, scans, symbols, trace, tracebacks
from . import history as blame  # history is also run_state's name for the session's earlier turns.
from .review import CHANGES as REVIEW_CHANGES, review, review_changes
from .test_runner import DEFAULT_IMAGE, command, run_snippet, run_tests
from .evidence import retain, discovery_task, check_answer, inventory
from .model import PROMPT_BUDGET, InvalidReply, RepeatedAnswer, compact_messages, read_limit
from .workspace import line_range, sections
from .prompts import (  # noqa: F401  (also imported from here by the tests and other modules)
    ACTIONS, CODE_ACTION, FIX_ANSWER, PYTHON_GUIDE, ANSWER_FROM_PRELOAD, ANSWER_WITH_POINTS, BRIEF_ANSWER, CHANGES_ANSWER, CHANGE_ADVICE, DISCOVERY_SYSTEM, EDIT_REQUEST,
    ENTRY_POINT_GUIDANCE, FINAL_ANSWER_CHARS, FINISH_NOW, POLITE, PRELOAD_FILES, QUESTION, RELOAD_SECONDS, REVIEW,
    SCHEMA, SYSTEM, POINTS, POINT_CHARS, OVERVIEW_CHARS,
)
from .tasks import (  # noqa: F401  (also imported from here by the tests and other modules)
    without_paths, entry_point_task, task_kind, code_names, focus_lines, search_named, section_note, step_label,
    page_note, named_files,
)
from .answers import (  # noqa: F401  (also imported from here by the tests and other modules)
    trim_at_limit, reply_problem, cited_schema, retrieved_rows, seen_span, file_named, with_points, action_key,
    format_search, format_retrieval, format_metrics, REPEAT_NOTE, LIMIT_NOTE, CORRECTION, SHOWN_ROWS,
    cite_mentions, MENTION_HEADING, check_values, VALUE_HEADING, missing_code, MISSING_HEADING, with_actions,
)


class AgentState(TypedDict):
    """State passed between the graph's nodes; a node returns only the keys it changes.

    messages          the whole conversation (compacted only when sent to the model)
    reply             the model's latest action; action "retry" marks a reply that was invalid
    steps             model calls made so far
    exit_code         0 finished, 2 incomplete or unverified
    repeated_lookup   the model repeated an action, so its next call must be the final answer
    search_evidence   lines seen so far (searches, reads, retrieval), which answers are checked against
    evidence_limited  some evidence was dropped or a tool hit a limit, so coverage is partial
    discovery         the FastAPI entry-point search already ran; the model only summarizes it
    invalid_replies   1 after an invalid reply; a second in a row stops the run
    cite_files        preloaded files that the points of an answer may cite
    kind              task_kind of the task: question, change, or other
    final_text        the answer as shown to the user; issues lists the answer checks it failed
    retried           the first answer, its text, and its issues, while the model is asked once to fix that answer
    """
    messages: list[dict[str, str]]
    reply: dict[str, str]
    steps: int
    exit_code: int
    repeated_lookup: bool
    search_evidence: list[dict]
    evidence_limited: bool
    discovery: bool
    task: str
    invalid_replies: int
    cite_files: list[str]
    kind: str
    final_text: str
    issues: list[str]
    retried: dict


def build_graph(model, workspace, *, max_steps=12, confirm_tests=None, knowledge=None, test_options=None, output=print, stream=None,
                confirm_code=None, retry_unverified=False):
    """Compile a per-task graph. Read-only: no node writes files. No remote services or persistent checkpoint store.

    confirm_code, a function shown the code, offers the run_python action; without it the action does not exist.
    retry_unverified asks the model once to fix an answer that fails its checks, and keeps whichever passes more.
    """
    if not 1 <= max_steps <= 30:
        raise ValueError("max_steps must be between 1 and 30")
    actions = ACTIONS + ((CODE_ACTION,) if confirm_code else ())
    write_stream = stream or (lambda _text: None)
    # A one-item list so call_model can set it: the reload warning is printed once per task.
    reload_warned = [False]

    def already_tried(state, reply):
        """True when this exact action already ran and its result is still in the model's view."""
        key = action_key(reply)
        try:
            visible = compact_messages(state["messages"], getattr(model, "prompt_budget", PROMPT_BUDGET))
        except ValueError:
            visible = state["messages"]
        for index in range(len(visible) - 1, -1, -1):
            message = visible[index]
            if message["role"] != "assistant":
                continue
            try:
                earlier = json.loads(message["content"])
            except json.JSONDecodeError:
                continue
            if isinstance(earlier, dict) and action_key(earlier) == key:
                following = visible[index + 1]["content"] if index + 1 < len(visible) else ""
                # A result compacted away may be requested again; the stub tells the model to do exactly that.
                return "[Older result compacted" not in following
        return False

    def call_model(state: AgentState):
        """Model node: ask for the next action, validate and announce it; a finish reply is also checked and shown."""
        step = state["steps"] + 1
        # Phase: choose the schema and messages. The model must answer now when it repeated an action, this is the
        # last allowed step, or discovery has already gathered the evidence.
        finishing = state.get("repeated_lookup", False) or step >= max_steps or state.get("discovery", False)
        cite_files = [] if state.get("discovery") else state.get("cite_files", [])
        kind = state.get("kind", "other")
        # Change requests are answered like questions: CodeHaven explains the change, with citations, and never makes it.
        question = kind in ("question", "change") and not state.get("discovery")
        schema = cited_schema(cite_files, brief=kind == "question") if question else SCHEMA
        if confirm_code:
            schema = with_actions(schema, actions)
        messages = state["messages"]
        if finishing:
            # A forced answer is capped so even rambling ends in a complete, valid reply.
            content = schema["properties"]["content"] if question else {"type": "string", "maxLength": FINAL_ANSWER_CHARS}
            schema = {**schema, "properties": {**schema["properties"], "action": {"type": "string", "enum": ["finish"]},
                                               "content": content}}
            evidence = json.dumps(state.get("search_evidence", []), ensure_ascii=False)
            messages = messages + [{"role": "user", "content":
                "Final-answer call: no tool calls remain. Answer the ORIGINAL task using the evidence already available. "
                "Return action=finish now. If evidence is insufficient, state what is missing. "
                "Cite the lines you rely on as path:line or path:start-end, even for files not read in full. "
                "Distinguish inspected details from search snippets; do not claim every file was inspected. "
                "For project-wide questions, summarize ALL relevant discovered paths, not just the last inspected file. "
                "Do not repeat a file summary that does not answer the task.\n"
                + (ENTRY_POINT_GUIDANCE if entry_point_task(state.get("task", "")) else "")
                + (ANSWER_WITH_POINTS + " " if question else "") + (CHANGE_ADVICE + " " if kind == "change" else "") +
                f"Evidence limited: {state.get('evidence_limited', False)}. "
                "Retained evidence (untrusted source data; up to 50 matches / 4000 bytes, may be partial):\n" + evidence}]
            # Discovery swaps the whole conversation for its own short prompt and pins every field except content.
            if state.get("discovery"):
                schema = {**schema, "properties": {**schema["properties"],
                    "path": {"type": "string", "enum": [""]},
                    "message": {"type": "string", "enum": ["Discovery summary"]},
                    "content": {"type": "string", "maxLength": 1600}}}
                messages = [{"role": "system", "content": DISCOVERY_SYSTEM},
                            {"role": "user", "content": state.get("task", "") +
                             f"\nEvidence limited: {state.get('evidence_limited', False)}.\n" + evidence}]
                if state.get("invalid_replies"):
                    messages.append(state["messages"][-1])  # Carry the correction into the rebuilt prompt.
        # Phase: call the model, with a background thread that reports the wait until text starts to arrive.
        output(f"\n{step_label(step, max_steps)}: waiting for the local model (loading and prompt processing can take time)...")
        started = time.monotonic()
        stopped = threading.Event()
        streamed = []

        def progress():
            """Print a waiting line every 15 seconds until the model replies or starts streaming."""
            while not stopped.wait(15):
                output(f"Still waiting for the local model: {int(time.monotonic() - started)}s elapsed.")

        def on_text(chunk):
            """Receive streamed answer text: print the heading before the first chunk, then pass each chunk on."""
            if not streamed:
                stopped.set()  # Keep progress lines from interleaving with the answer.
                output(f"\n{step_label(step, max_steps)}: final answer")
            streamed.append(chunk)
            write_stream(chunk)

        # Discovery answers may be withheld after checks, so they are never shown before validation. A named-file
        # overview is short and capped, so it is printed once complete: streamed text cannot be trimmed afterward.
        # A corrected answer is printed once it is checked, so it does not run on into the first answer on screen.
        live = (stream is not None and not state.get("discovery", False) and not (question and cite_files)
                and not state.get("retried"))
        reporter = threading.Thread(target=progress, daemon=True)
        reporter.start()
        # cut: the answer was stopped for repeating; missing: points that name nothing in their file;
        # capped: the answer hit its length limit; overview: the answer text before points were folded in.
        cut, missing, capped, overview = False, [], False, ""
        try:
            try:
                reply = model.chat(messages, schema, on_text=on_text) if live else model.chat(messages, schema)
                if isinstance(reply, dict) and reply.get("action") == "finish" and isinstance(reply.get("content"), str):
                    reply["content"], capped = trim_at_limit(reply["content"], schema["properties"]["content"].get("maxLength"))
                    overview = reply["content"]
                if question:
                    reply, missing = with_points(reply, workspace, state.get("search_evidence", []))
                problem = reply_problem(reply, actions)
            except RepeatedAnswer as exc:
                # Keep the useful text before the loop; a retry at temperature 0 tends to loop again.
                reply, problem, cut = (dict(action="finish", path="", content=exc.answer, message="Answer stopped while repeating"),
                                       None, True) if exc.answer else (None, str(exc), False)
            except InvalidReply as exc:
                reply, problem = None, str(exc)
        finally:
            stopped.set()
            reporter.join()
            if streamed:
                write_stream("\n")
            metrics = getattr(model, "last_metrics", None)
            if isinstance(metrics, dict) and metrics:
                output(format_metrics(metrics))
                if step > 1 and not reload_warned[0] and (metrics.get("load_seconds") or 0) >= RELOAD_SECONDS:
                    reload_warned[0] = True
                    output(f"Warning: Ollama reloaded the model between steps ({metrics['load_seconds']:.0f}s). Something else is "
                           "probably using Ollama and competing for memory; check with `ollama ps`. Steps will stay slow until it stops.")
        steps = state["steps"] + 1
        # Phase: handle an unusable reply. The first one gets a correction and another try; a second in a row ends the run.
        if problem:
            if state.get("invalid_replies", 0):
                raise ValueError(f"Model returned an invalid action twice in a row: {problem}")
            output(f"\n{step_label(steps, max_steps)}: invalid model reply ({problem}); asking the model to correct it.")
            return {"messages": state["messages"] + [{"role": "user", "content": CORRECTION.format(problem)}],
                    "reply": {"action": "retry", "path": "", "content": "", "message": ""},
                    "steps": steps, "invalid_replies": 1}
        assert isinstance(reply, dict)
        # A tool requested on a final-answer call is not run: exit code 2 makes route_action end the graph.
        if finishing and reply["action"] != "finish":
            output("Final-answer request failed: the model requested another tool. The task is incomplete; no further tool was executed.")
            return {"reply": reply, "steps": steps, "exit_code": 2, "repeated_lookup": True, "invalid_replies": 0}
        repeated = reply["action"] != "finish" and already_tried(state, reply)
        # Phase: announce the action, describing its target in the action's own terms.
        if reply["action"] == "search":
            target = f"query={json.dumps(reply['content'])} scope={json.dumps(reply['path'] or '.') }"
        elif reply["action"] == "read" and isinstance(span := action_key(reply)[2], tuple):
            target = f"{reply['path']} lines {span[0]}-{span[1] or 'end'}"
        else:
            target = reply["path"] or (reply["content"][:100] if reply["action"] == "retrieve" else "")
        shown = "".join(streamed)
        if reply["action"] == "finish":
            if not streamed:
                output(f"\n{step_label(steps, max_steps)}: final answer")
        else:
            output(f"\n{step_label(steps, max_steps)}: {reply['action']} {target} ({reply['message']})")
        if repeated:
            output("Repeated action detected; skipping the duplicate tool call and requesting a final answer.")
        # Phase: check a final answer against the evidence, and print whatever part of it was not already streamed.
        final_text, issues = None, []
        if reply["action"] == "finish":
            invented = []
            # Each literal value the answer reports is looked for in the code: on the cited lines, or else on a line the
            # model was shown, which then gets a citation; a value on no shown line was not read from the code.
            if question and reply.get("content") and not state.get("discovery"):
                extra, invented = check_values(reply["content"], state.get("task", ""), workspace,
                                               state.get("search_evidence", []))
                if extra:
                    reply["content"] = "\n\n".join([reply["content"].rstrip(), VALUE_HEADING + "\n" + "\n".join(extra)])
                    output(f"CodeHaven cited {len(extra)} line(s) holding values the answer reports.")
            issues = check_answer(reply["content"] or reply["message"], state.get("search_evidence", []),
                                  discovery=state.get("discovery", False), require_citation=question)
            # An answer that cites nothing gets citations for the code names it mentions, from lines the model saw.
            if question and any(issue.startswith("Repository claims require") for issue in issues) and reply.get("content"):
                sources = cite_mentions(reply["content"], workspace, state.get("search_evidence", []))
                if sources:
                    reply["content"] = "\n\n".join([reply["content"].rstrip(), MENTION_HEADING + "\n" + "\n".join(sources)])
                    output(f"CodeHaven added {len(sources)} citation(s) for code names the answer mentions.")
                    issues = check_answer(reply["content"], state.get("search_evidence", []),
                                          discovery=state.get("discovery", False), require_citation=question)
            issues += [f"No part named {symbol!r} was found in {path}; that point is uncited." for path, symbol in missing]
            issues += [f"The answer reports {value!r}, which is on no line the model was shown." for value in invented]
            # Said after the checks, which it takes no part in: the cited code calls into code that is not here, so the
            # reader knows the answer cannot show what that code does.
            if question and reply.get("content") and not state.get("discovery"):
                outside = missing_code(reply["content"], workspace, knowledge)
                if outside:
                    reply["content"] = "\n\n".join([reply["content"].rstrip(), MISSING_HEADING + "\n" + "\n".join(outside)])
                    output(f"CodeHaven noted {len(outside)} import(s) of code that is not in this repository.")
            if issues and state.get("discovery"):
                output("Model summary withheld because answer checks failed.")
                final_text = inventory(state.get("search_evidence", []), state.get("evidence_limited", False))
                output(final_text)
            else:
                if cut:
                    output(REPEAT_NOTE)
                # Print unless already streamed; a streamed loop is followed by the note, not a second copy.
                answer = reply["content"] or reply["message"]
                final_text = "\n".join([answer] + ([REPEAT_NOTE] if cut else []) + ([LIMIT_NOTE] if capped else []))
                if capped and streamed:
                    # The unfinished sentence was already on screen; say it is dropped and show only the points after it.
                    output(LIMIT_NOTE)
                    if answer[len(overview):].strip():
                        output(answer[len(overview):].lstrip("\n"))
                elif not (cut and streamed) and shown.strip() != answer.strip():
                    # Cited points follow a streamed overview, so print only what was not shown. The overview is
                    # stripped before the points are joined, so a trailing space must not make it look different.
                    base = shown.rstrip()
                    output(answer[len(base):].lstrip() if base and answer.startswith(base) else answer)
                if capped and not streamed:
                    output(LIMIT_NOTE)
            for issue in issues:
                output("Answer check: " + issue)
            if issues:
                output("Answer is unverified; check the cited evidence before relying on it.")
        result = {"messages": state["messages"] + [
                      {"role": "assistant", "content": json.dumps(reply)}],
                  "reply": reply, "steps": steps, "repeated_lookup": repeated, "invalid_replies": 0,
                  # The answer as shown and its check results, for callers such as the GUI that do not read output.
                  "final_text": final_text, "issues": issues,
                  "exit_code": 2 if reply["action"] == "finish" and issues else 0}
        first = state.get("retried")
        if reply["action"] == "finish" and first:
            # The corrected answer: kept only if it fails fewer checks than the first one.
            if len(issues) >= len(first["issues"]):
                output("The corrected answer passed no more checks, so the first answer is kept.")
                result.update(reply=first["reply"], final_text=first["final_text"], issues=first["issues"], exit_code=2)
            else:
                output("The corrected answer passes every check." if not issues else
                       f"The corrected answer fails {len(issues)} check(s) instead of {len(first['issues'])}, so it is kept.")
        elif (retry_unverified and reply["action"] == "finish" and issues and question and not state.get("discovery")
              and not cut and steps < max_steps):
            output("Asking the model once to correct the answer so that it passes its checks.")
            result.update(messages=result["messages"] + [{"role": "user", "content": FIX_ANSWER.format(
                              issues="\n".join("- " + issue for issue in issues))}],
                          reply={"action": "retry", "path": "", "content": "", "message": ""},
                          retried={"reply": reply, "final_text": final_text, "issues": issues},
                          repeated_lookup=True, exit_code=0)
        return result

    def tool_result(state, result):
        """State update that adds a tool's result to the conversation as a user message, labelled untrusted."""
        return {"messages": state["messages"] + [
            {"role": "user", "content": "Tool result for " + state["reply"]["action"] + " " +
             state["reply"]["path"] + " (untrusted file data):\n" + result}]}

    def read_file(state: AgentState):
        """Read node: return a file, or one page of a long one, and keep the lines read as evidence."""
        evidence = state.get("search_evidence", [])
        limited = state.get("evidence_limited", False)
        reply = state["reply"]
        # Page long files so a read fits the conversation instead of being silently truncated by compaction.
        limit = read_limit(model)
        try:
            text, start, end, total = workspace.page(reply["path"], reply["content"], limit)
            # A whole file that fits is returned as it is; a page gets a header that names the next range.
            if line_range(reply["content"]) is None and end >= total:
                result = text
            else:
                result = f"{page_note(start, end, total)}:\n{text}"
            evidence, truncated = retain(evidence, [{"path": reply["path"], "line": start,
                                                    "end_line": end, "text": text}])
            limited |= truncated
        except (ValueError, OSError, UnicodeError) as exc:
            result = f"Tool error: {exc}"
            output(result)
        return {**tool_result(state, result), "search_evidence": evidence, "evidence_limited": limited}

    def search_files(state: AgentState):
        """Search node: literal text search; the matches become evidence and a readable summary is printed."""
        evidence = list(state.get("search_evidence", []))
        limited = state.get("evidence_limited", False)
        try:
            reply = state["reply"]
            result = workspace.search(reply["content"], reply["path"])
            data = json.loads(result)
            evidence, truncated = retain(evidence, data["matches"])
            limited |= truncated or data.get("limit_reached", False) or data.get("candidate_limit_reached", False)
            output(format_search(data))
        except (ValueError, OSError, UnicodeError) as exc:
            result = f"Tool error: {exc}"
            output(result)
        return {**tool_result(state, result), "search_evidence": evidence, "evidence_limited": limited}

    def retrieve_references(state: AgentState):
        """Retrieve node: query the knowledge base (indexed code, docs, user references) and keep the passages as evidence."""
        reply = state["reply"]
        evidence = state.get("search_evidence", [])
        limited = state.get("evidence_limited", False)
        try:
            if reply["path"]:
                raise ValueError("retrieve uses the configured knowledge base; path must be empty")
            if knowledge is None:
                raise ValueError("No knowledge base is configured")
            result = knowledge.retrieve(reply["content"])
            data = json.loads(result)
            evidence, truncated = retain(evidence, retrieved_rows(data))
            limited |= truncated or data.get("budget_limited", False) or data.get("results_limited", False)
            output(format_retrieval(data))
        except (ValueError, OSError, UnicodeError) as exc:
            result = f"Tool error: {exc}"
            output(result)
        return {**tool_result(state, result), "search_evidence": evidence, "evidence_limited": limited}

    def execute_tests(state: AgentState):
        """Test node: run the configured tests in Docker, only after the user approves the command shown."""
        reply = state["reply"]
        if reply["path"] or reply["content"]:
            return tool_result(state, "Tool error: run_tests accepts no path, content, or custom command")
        options = dict(test_options or {})
        output("Requested test command: " + " ".join(command(options.get("runner", "unittest"))))
        output("Isolation: Docker; network disabled; filtered project snapshot read-only. "
               "Image: " + options.get("image", DEFAULT_IMAGE))
        output(f"Working directory: {workspace.root}")
        # With no way to ask the user, the run counts as denied.
        if not confirm_tests or not confirm_tests():
            return tool_result(state, "Tests were NOT run: execution was not approved.")
        try:
            result = run_tests(workspace, **options)
        except (ValueError, OSError) as exc:
            result = f"Test execution error: {exc}"
        output(result)
        return tool_result(state, result)

    def execute_code(state: AgentState):
        """Code node: run the model's Python snippet in the test sandbox, only after the user approves the code shown."""
        reply = state["reply"]
        if reply["path"]:
            return tool_result(state, "Tool error: run_python takes the code in content and an empty path")
        code = reply["content"]
        output("Requested Python snippet (runs in Docker with network disabled and a read-only copy of the project):\n"
               + code.rstrip())
        if not confirm_code(code):
            return tool_result(state, "The snippet was NOT run: the user did not approve it. Answer from the code instead.")
        try:
            result = run_snippet(workspace, code, **dict(test_options or {}))
        except (ValueError, OSError) as exc:
            result = f"Snippet error: {exc}"
        output(result)
        return tool_result(state, result)

    def route_action(state: AgentState):
        """Edge after the model node: choose the next node from the reply."""
        # An invalid reply was just sent back for correction: call the model again, unless the steps ran out.
        if state["reply"]["action"] == "retry":
            return after_tool(state)
        if state["reply"]["action"] == "finish":
            return "finish"
        # The model asked for a tool on a final-answer call; stop without running it.
        if state.get("exit_code") == 2:
            return "finish"
        if state.get("repeated_lookup"):
            return "repeat"
        return state["reply"]["action"]

    def repeated_lookup(state: AgentState):
        """Repeat node: stands in for a duplicate tool call and tells the model to answer instead."""
        return tool_result(state, "Duplicate action skipped: this exact request already ran and its result is above. "
                           "Answer the original task or explain the evidence gap.")

    def after_tool(state: AgentState):
        """Edge after every tool node: back to the model, or to limit when the steps ran out."""
        return "limit" if state["steps"] >= max_steps else "model"

    def limit(state: AgentState):
        """Limit node: report that the steps ran out and mark the task incomplete."""
        output("Step limit reached without a final answer. The task is incomplete; tool messages above are not a final answer.")
        return {"exit_code": 2}

    def discover(state: AgentState):
        """Discover node: run the fixed FastAPI( search up front, so the model only has to summarize the matches."""
        output("Plan: search broadly for FastAPI constructors, classify candidates, then summarize with citations.")
        data = json.loads(workspace.search("FastAPI("))
        evidence, limited = retain([], data["matches"])
        limited |= data["limit_reached"] or data["candidate_limit_reached"]
        output(f"Discovery: {len(data['matches'])} matches across {data['files_searched']} files; limited={limited}.")
        return {"search_evidence": evidence, "evidence_limited": limited, "discovery": True}

    # Nodes: the model, one node per tool action, and three helpers (limit, repeat, discover).
    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("read", read_file)
    graph.add_node("search", search_files)
    graph.add_node("retrieve", retrieve_references)
    graph.add_node("run_tests", execute_tests)
    graph.add_node(CODE_ACTION, execute_code)
    graph.add_node("limit", limit)
    graph.add_node("repeat", repeated_lookup)
    graph.add_node("discover", discover)
    # Entry: an explicit FastAPI entry-point task searches first; every other task starts at the model.
    graph.add_conditional_edges(START, lambda state: "discover" if discovery_task(state.get("task", "")) else "model")
    graph.add_edge("discover", "model")
    # After the model: run the chosen tool, end on finish, or return to the model after a corrected reply.
    graph.add_conditional_edges("model", route_action,
                                {"read": "read", "search": "search", "retrieve": "retrieve",
                                 "run_tests": "run_tests", CODE_ACTION: CODE_ACTION, "repeat": "repeat", "finish": END,
                                 "model": "model", "limit": "limit"})
    # After a tool: its result goes back to the model, or to limit when no steps remain.
    for tool in ("read", "search", "retrieve", "run_tests", CODE_ACTION, "repeat"):
        graph.add_conditional_edges(tool, after_tool, {"model": "model", "limit": "limit"})
    graph.add_edge("limit", END)
    return graph.compile()


def run(model, workspace, task, *, memory=None, **options):
    """Run one task and return its exit code: 0 finished, 2 incomplete or unverified.

    With a Memory, saved answers from earlier sessions are recalled and a checked answer is saved.
    """
    if memory is not None:
        return Session(model, workspace, memory=memory, **options).ask(task)
    return run_state(model, workspace, task, **options)["exit_code"]


def run_state(model, workspace, task, *, max_steps=12, confirm_tests=None, knowledge=None,
              test_options=None, output=print, stream=None, history="", carried_evidence=(), focus_file=None, notes="",
              confirm_code=None, retry_unverified=False, editor=None):
    """Run one task and return the final graph state.

    editor, a coder.Editor, turns on changing code: a change request is then made, after the user approves its diff. history and carried_evidence come from a Session.

    notes are answers recalled from long-term memory, shown before the session's own history.

    focus_file names the file every task is about, such as a file uploaded in the GUI: it is preloaded even
    when the task does not name it.
    """
    # Phase 1: the opening messages: the system prompt, then the task with the file paths most relevant to it.
    files = workspace.files()
    terms = set(re.findall(r"[a-z0-9]+", task.lower()))
    def relevance(name):
        """Sort key: paths sharing the most words with the task first, then shallower paths, then by name."""
        words = set(re.findall(r"[a-z0-9]+", name.lower()))
        return (-len(terms & words), name.count("/"), name)
    selected = []
    size = 0
    # At most 40 paths and 2000 bytes, so the listing never crowds the code out of the prompt.
    for name in sorted(files, key=relevance):
        if len(selected) >= 40 or size + len(name.encode()) + 1 > 2000:
            break
        selected.append(name)
        size += len(name.encode()) + 1
    earlier = ((f"Notes saved in earlier sessions about this repository (the files they cite have not changed since; "
                f"use them only where they help with the current task):\n{notes}\n\n" if notes else "")
               + (f"Earlier in this session (answers may be outdated if files changed since):\n{history}\n\n" if history else ""))
    earlier += "Current task: " if earlier else ""
    messages = [{"role": "system", "content": SYSTEM + ("\n" + PYTHON_GUIDE if confirm_code else "")},
                {"role": "user", "content": earlier + task + f"\nWorkspace path sample ({len(selected)} of {len(files)} listed files; use search/read/retrieve for more):\n" +
                 "\n".join(selected)}]
    evidence, evidence_limited = [dict(row) for row in carried_evidence], False
    # Phase 2: a task about git changes gets them read with git up front.
    asked_changes = changes.requested(task)
    report = found = None
    if (written := commit_messages.requested(task)):
        # "Write a commit message": the diff goes to the model with a prompt of its own, and nothing is committed.
        found = commit_messages.write(model, workspace, task, written, read_limit(model) - 600, output)
    elif asked_changes and REVIEW_CHANGES.search(task):
        # "Review my changes": only the functions the diff touched are reviewed, one model call each.
        try:
            found = review_changes(model, workspace, asked_changes, spans=knowledge.dependencies.spans if knowledge else None,
                                   output=output, in_arabic=arabic.written_in(task))
        except changes.NoChanges as exc:
            found = {"text": str(exc), "evidence": []}
        except (ValueError, OSError) as exc:
            found = {"text": f"Could not read the changes with git ({exc}).", "evidence": []}
    elif asked_changes:
        try:
            report = changes.describe(workspace, asked_changes, read_limit(model) - len(CHANGES_ANSWER.encode()) - 200,
                                      named_files(task, files))
        except changes.NoChanges as exc:
            # Nothing to explain is a fact git reports, so it is answered at once, without the model.
            output(str(exc))
            return {"messages": messages, "reply": {"action": "finish", "path": "", "content": str(exc), "message": ""},
                    "steps": 0, "exit_code": 0, "task": task, "search_evidence": evidence, "evidence_limited": False,
                    "final_text": str(exc), "issues": [], "kind": "question"}
        except (ValueError, OSError) as exc:
            output(f"Could not read the changes with git ({exc}); answering from the files instead.")
    # Phase 3: shortcuts that return without running the graph: a file review, a call trace, test gaps, or a project tour.
    # Some requests have a purpose-built answer that does not go through the general workflow.
    reviewed = named_files(task, files)[:1] or ([focus_file] if focus_file in files else [])
    wanted = insights.requested(task) if report is None and not asked_changes else None
    in_arabic = arabic.written_in(task)
    scan = scans.requested(task) if report is None and not asked_changes else None
    why = report is None and not asked_changes and blame.requested(task)
    edit = coder.requested(task) if editor is not None and report is None and found is None else None
    if found is not None:
        pass  # The changes were reviewed above.
    elif edit:
        found = coder.run(edit, model, workspace, knowledge, task, editor, output)
    elif scan:
        found = scans.run(*scan, workspace, knowledge)
    elif report is None and not asked_changes and (guide := guides.requested(task)):
        found = guides.run(*guide, model, workspace, knowledge, output)
    elif why and (explained := blame.explain(model, workspace, knowledge, task, output)) is not None:
        found = explained
    elif report is None and (REVIEW.search(task) or arabic.REVIEW.search(arabic.normalise(task))) and reviewed:
        spans = knowledge.dependencies.spans.get(reviewed[0]) if knowledge is not None else None
        try:
            found = review(model, workspace, reviewed[0], spans=spans, output=output, in_arabic=in_arabic)
        except (ValueError, OSError, UnicodeError) as exc:
            found = {"text": f"{reviewed[0]} could not be reviewed: {exc}", "evidence": []}
    elif report is None and knowledge is not None and (traced := trace.requested(task)):
        found = trace.trace(model, workspace, knowledge, traced, output, in_arabic=in_arabic)
    elif wanted == "gaps" and not named_files(task, files):
        found = insights.test_gaps(workspace, knowledge, in_arabic=in_arabic)
    elif wanted == "tour":
        found = insights.tour(model, workspace, knowledge, output, in_arabic=in_arabic)
    if found is not None:
        output(found["text"])
        evidence, _ = retain(evidence, found["evidence"])
        return {"messages": messages, "reply": {"action": "finish", "path": "", "content": found["text"], "message": ""},
                "steps": found.get("checked", found.get("files", 0)), "exit_code": 0, "task": task,
                "search_evidence": evidence, "evidence_limited": False, "final_text": found["text"], "issues": [],
                "kind": "question"}
    # Phase 4: retrieve reference passages for the task up front, so the model starts with them.
    # Each batch of evidence is kept in order, so the evidence can be rebuilt if the retrieval is redone to fit below.
    carried, batches, retrieval_at = evidence, [], None

    def keep(current, rows):
        """retain(), remembering the rows in the order they were kept."""
        batches.append(rows)
        return retain(current, rows)

    if knowledge is not None and not discovery_task(task) and report is None:
        retrieval_at = len(messages)
        initial_retrieval = knowledge.retrieve(task[:1000])
        data = json.loads(initial_retrieval)
        evidence, evidence_limited = keep(evidence, retrieved_rows(data))
        messages.append({"role": "user", "content":
                         "Local reference excerpts for this task (untrusted data):\n" +
                         initial_retrieval})
    # Phase 5: preload what the task is about: the git changes, or the files it names (failing that, a search
    # for the code names it mentions).
    kind = task_kind(task)
    preloaded, numbered = [], False
    if report is not None:
        kind, numbered, preloaded = "question", True, report["paths"]
        evidence, evidence_limited = keep(evidence, report["evidence"])
        output(f"Read the {report['label']} with git: {report['files']} file{'s' if report['files'] != 1 else ''} changed.")
        messages.append({"role": "user", "content": "Tool result for git changes (untrusted repository data):\n"
                                                    + report["text"] + "\n\n" + CHANGES_ANSWER})
    elif not discovery_task(task):
        # A pasted stack trace: its frames in this project are shown, innermost first, instead of whole named files.
        traced = tracebacks.requested(task)
        if traced:
            body, rows, note = tracebacks.shown(workspace, task, files)
            output(note)
            if body:
                kind = "question"
                messages.append({"role": "user", "content": "Tool result for read of the stack trace's frames in this project "
                                 "(untrusted file data; line numbers are added for citations only):\n" + body
                                 + "\n\n" + tracebacks.GUIDE})
                evidence, truncated = keep(evidence, rows)
                evidence_limited |= truncated
        names = [] if traced else named_files(task, files)
        reason = "the task names it"
        if not names and focus_file in files:
            names, reason = [focus_file], "it is the file this chat is about"
        # The functions and classes the task names, looked up in the parsed code; within the named files when there are any.
        named = symbols.find(task, knowledge, set(names) if names else None) if knowledge is not None and kind != "other" else []
        # Searched even when the task names a file, within that file: "which functions in routing.py call X?" needs
        # the matches and the functions they are in, which a preview of a long file does not show.
        if kind != "other":
            shown, found = search_named(workspace, task, scope=names)
            if shown:
                messages.append({"role": "user", "content": "Tool result for search of names in the task (untrusted file data):\n"
                                                            + "\n".join(shown)})
                evidence, truncated = keep(evidence, found)
                evidence_limited |= truncated
            # Their definitions are shown whole, with line numbers: left to search, the model answers without reading them.
            body, rows, notes = symbols.shown(workspace, named, task, knowledge=knowledge)
            if body:
                for note in notes:
                    output(note)
                messages.append({"role": "user", "content": "Tool result for read of the definitions the task names (untrusted file data; "
                                                            "line numbers are added for citations only; do not read these lines again):\n" + body})
                evidence, truncated = keep(evidence, rows)
                evidence_limited |= truncated
        # Line numbers help citations; nothing is ever edited, so files need not stay byte-exact.
        numbered = kind in ("question", "change")
        guide = FINISH_NOW + ANSWER_FROM_PRELOAD
        # The byte budget is shared between the named files, less the room the closing guide needs.
        limit = (read_limit(model) - (len(guide.encode()) + 2 if numbered else 0)) // max(1, len(names))
        for name in names:
            try:
                text = workspace.read(name)
            except (ValueError, OSError, UnicodeError):
                continue
            lines = text.splitlines(keepends=True)
            total = len(lines)
            # A file too long for its share is shown in parts; 6 is the bytes each line-number prefix adds.
            # A file too long to show whole keeps the lines of the parts the task names, not only its start and end.
            focus = focus_lines(task, text) + [line for path, _, start, end in named if path == name for line in range(start, end + 1)]
            ranges = sections(text, limit, focus, 6 if numbered else 0)
            parts, shown_ranges = [], []
            for index, (start, end) in enumerate(ranges):
                chosen = lines[start - 1:end]
                body = "".join(f"{start + offset:>4}| {line}" for offset, line in enumerate(chosen)) if numbered else "".join(chosen)
                following = ranges[index + 1][0] if index + 1 < len(ranges) else None
                parts.append(f"{section_note(start, end, following, total)}:\n{body}")
                shown_ranges.append(f"{start}-{end}")
                evidence, truncated = keep(evidence, [{"path": name, "line": start, "end_line": end, "text": "".join(chosen)}])
                evidence_limited |= truncated
            output(f"Preloaded {name} (lines {' and '.join(shown_ranges)} of {total}) because {reason}.")
            messages.append({"role": "user", "content":
                             f"Tool result for read {name} (untrusted file data, preloaded because {reason}; "
                             "do not read these lines again):\n"
                             # An explicit citation example here made the 7B model fetch small ranges instead of answering.
                             + ("Line numbers are added for citations only and are not part of the file.\n" if numbered else "")
                             + "\n".join(parts)})
            preloaded.append(name)
        # Joined to the last message, not sent alone: compaction keeps only the newest message whole.
        advice = " " + CHANGE_ADVICE if kind == "change" else ""
        if numbered and preloaded:
            messages[-1] = {**messages[-1], "content": messages[-1]["content"] + "\n\n" + guide + advice}
        elif numbered:
            brief = BRIEF_ANSWER if kind == "question" else ""
            messages[-1] = {**messages[-1], "content": messages[-1]["content"] + "\n\n" + ANSWER_WITH_POINTS + brief + advice}
    # Joined to the last message for the same reason. English tasks get nothing added, so their prompts are unchanged.
    if in_arabic:
        messages[-1] = {**messages[-1], "content": messages[-1]["content"] + "\n\n" + arabic.ANSWER_IN_ARABIC}
    # When the preloaded files leave too little room, compaction would replace the reference excerpts with a stub, so
    # they were found for nothing and the model was told to fetch them again. Retrieve again to fit the room left, or
    # leave them out when not even a small retrieval fits. The newest message is never stubbed, so it is left alone.
    budget = getattr(model, "prompt_budget", PROMPT_BUDGET)
    total = sum(len(message["content"].encode()) + 64 for message in messages)
    if retrieval_at is not None and retrieval_at < len(messages) - 1 and total > budget:
        header = messages[retrieval_at]["content"].split("\n", 1)[0] + "\n"
        room = budget - (total - len(messages[retrieval_at]["content"].encode())) - len(header.encode()) - 200
        if room >= 2000:
            fitted = knowledge.retrieve(task[:1000], budget=room)
            batches[0] = retrieved_rows(json.loads(fitted))
            messages[retrieval_at] = {**messages[retrieval_at], "content": header + fitted}
        else:
            batches[0] = []
            del messages[retrieval_at]
        # The model is shown only the new excerpts, so only those count as lines it has seen.
        evidence, evidence_limited = carried, False
        for rows in batches:
            evidence, truncated = retain(evidence, rows)
            evidence_limited |= truncated
    # Phase 6: run the graph. A step is at most two graph hops (the model, then a tool), plus discover and limit,
    # which gives the recursion limit below.
    # Always offline: LangChain would otherwise start remote tracing from LANGSMITH_TRACING and similar env vars.
    with tracing_context(enabled=False):
        graph = build_graph(model, workspace, max_steps=max_steps, confirm_tests=confirm_tests,
                            knowledge=knowledge, test_options=test_options, output=output, stream=stream,
                            confirm_code=confirm_code, retry_unverified=retry_unverified)
        return graph.invoke({"messages": messages, "reply": {}, "steps": 0, "exit_code": 0, "task": task,
                             "search_evidence": evidence, "evidence_limited": evidence_limited, "invalid_replies": 0,
                             "cite_files": preloaded if numbered else [], "kind": kind},
                            config={"recursion_limit": 2 * max_steps + 2, "callbacks": [],
                                    "run_name": "CodeHaven"})


class Session:
    """Several tasks in a row with one workspace, index, and model.

    Follow-ups see a bounded record of earlier questions and answers plus retained evidence,
    not full tool transcripts, so prompts stay small and the system-prompt prefix stays cacheable.

    Memory is tiered: the last RECENT answers are kept nearly whole, because follow-ups such as
    "and what does it return?" refer to them, while older ones keep only their question and first
    sentence. The whole record stays under HISTORY_BYTES, so it never crowds the code out of a
    12 KB prompt; the oldest entries go first.

    With a Memory (opt-in), answers that passed their checks are also saved to disk, and relevant ones whose
    cited files are unchanged are shown again in later sessions. saved tells whether the last answer was kept.
    """
    TURNS, RECENT = 8, 2
    QUESTION_BYTES, ANSWER_BYTES = 300, 1000
    OLDER_QUESTION_BYTES, OLDER_ANSWER_BYTES = 160, 200
    HISTORY_BYTES = 3200

    def __init__(self, model, workspace, *, knowledge=None, memory=None, **options):
        """options are passed on to run_state for every task (max_steps, output, stream, and so on)."""
        self.model, self.workspace, self.knowledge, self.options = model, workspace, knowledge, options
        self.memory, self.saved = memory, False
        self.reset()

    def reset(self):
        """Forget the conversation so far: its turns, its retained evidence, and the last graph state."""
        self.turns, self.evidence, self.last_state = [], [], {}

    @staticmethod
    def first_sentence(answer):
        """The opening of an answer, without the cited points listed after it."""
        line = next((line.strip() for line in answer.splitlines() if line.strip()), "")
        match = re.match(r"(.+?[.!?؟])(?:\s|$)", line)
        return match[1] if match else line

    def history(self):
        """The bounded record of earlier turns that run_state puts before the next task."""
        def clip(text, limit):
            """Shorten text to limit UTF-8 bytes without splitting a character."""
            raw = text.encode()
            return text if len(raw) <= limit else raw[:limit].decode("utf-8", "ignore") + " …"
        entries = []
        for index, (question, answer) in enumerate(self.turns):
            if index >= len(self.turns) - self.RECENT:
                entries.append(f"Q: {clip(question, self.QUESTION_BYTES)}\nA: {clip(answer, self.ANSWER_BYTES)}")
            else:
                entries.append(f"Q: {clip(question, self.OLDER_QUESTION_BYTES)}\n"
                               f"A (summary): {clip(self.first_sentence(answer), self.OLDER_ANSWER_BYTES)}")
        # Over budget: drop the oldest entries, always keeping the latest one.
        while len(entries) > 1 and len("\n".join(entries).encode()) > self.HISTORY_BYTES:
            entries.pop(0)
        return "\n".join(entries)

    def ask(self, task):
        """Run one task with this session's history and evidence, record its answer, and return the exit code."""
        if self.turns and self.knowledge is not None:
            self.knowledge.refresh()  # Pick up files changed between questions.
        # "My last commit" means something else tomorrow, so answers about changes are neither recalled nor saved.
        # Test gaps and tours are rebuilt from the whole project each time, which a saved copy would not notice changing.
        memory = (None if changes.requested(task) or insights.requested(task) or commit_messages.requested(task)
                  or guides.requested(task) or scans.requested(task) or blame.requested(task)
                  or (self.options.get("editor") is not None and coder.requested(task)) else self.memory)
        notes = memory.recall(task, skip=[question for question, _ in self.turns]) if memory else []
        evidence = self.evidence
        if notes:
            self.options.get("output", print)(f"Recalled {len(notes)} saved answer{'s' if len(notes) != 1 else ''} "
                                              "from earlier sessions; the files they cite have not changed.")
            evidence, _ = retain(evidence, self.memory.evidence(notes))
        state = run_state(self.model, self.workspace, task, knowledge=self.knowledge, history=self.history(),
                          carried_evidence=evidence, notes=self.memory.text(notes) if notes else "", **self.options)
        reply = state.get("reply") or {}
        answer = (reply.get("content") or reply.get("message") or "") if reply.get("action") == "finish" else ""
        self.turns = (self.turns + [(task, answer or "(no final answer)")])[-self.TURNS:]
        self.evidence = state.get("search_evidence", [])
        self.last_state = state
        # Only an answer that passed its checks is worth keeping for another day.
        self.saved = bool(memory and answer and state["exit_code"] == 0 and memory.add(task, answer))
        return state["exit_code"]
