"""Ollama transport: numeric loopback only, no proxies or redirects."""
import ipaddress
import json
import re
import urllib.error
import urllib.parse
import urllib.request

MODEL = "qwen2.5-coder:7b"  # The default, and the model the evals were tuned and measured on.
MODEL_NAME = re.compile(r"[\w.:/@-]{1,200}")


def cloud_name(name):
    """Ollama names cloud-backed models with a cloud tag; CodeHaven never sends code to them."""
    return name.endswith(("-cloud", ":cloud"))
MAX_RESPONSE_BYTES = 2_000_000
STUB_MIN_BYTES = 400


class InvalidReply(ValueError):
    """The model answered, but not with a usable action; the agent may ask it to correct itself."""


class RepeatedAnswer(InvalidReply):
    """A finish answer began looping and generation was stopped. answer holds the text before the loop."""

    def __init__(self, answer):
        super().__init__("the final answer kept repeating the same text, so generation was stopped; "
                         "answer briefly and state each point only once")
        self.answer = answer


class RepetitionGuard:
    """Stop a final answer that keeps repeating the same phrase, as small models can at temperature 0.

    Checked on every streamed piece, so the window slides across a loop's shared phrase even when
    the surrounding words drift. A global repeat penalty is avoided because edits must copy text exactly.
    """
    WINDOW, TIMES, DISTINCT = 60, 3, 8

    def __init__(self):
        self.text = ""

    def feed(self, piece):
        self.text += piece
        if len(self.text) < self.WINDOW * self.TIMES:
            return
        tail = self.text[-self.WINDOW:]
        # Separator runs such as "-----" are ignored; real loops have varied characters.
        if len(set(tail)) >= self.DISTINCT and self.text.count(tail) >= self.TIMES:
            raise RepeatedAnswer(self.before_loop(tail))

    def before_loop(self, tail):
        """Keep everything up to the loop's second pass, trimmed back to a sentence or line end."""
        first = self.text.find(tail)
        kept = self.text[:self.text.find(tail, first + len(tail))]
        end = max(kept.rfind(". "), kept.rfind(".\n"), kept.rfind("\n"))
        return (kept[:end + 1] if end > 0 else kept).strip()


class FinishContent:
    """Incrementally decode the content string of a streamed finish action and pass new text to emit."""
    ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}

    def __init__(self, emit):
        self.emit = emit
        self.buffer = ""
        self.position = None
        self.done = False

    def feed(self, text):
        self.buffer += text
        if self.done:
            return
        if self.position is None:
            content = re.search(r'"content"\s*:\s*"', self.buffer)
            if not content or not re.search(r'"action"\s*:\s*"finish"', self.buffer):
                return
            self.position = content.end()
        decoded, index, buffer = [], self.position, self.buffer
        while index < len(buffer):
            char = buffer[index]
            if char == '"':
                self.done = True
                break
            if char != "\\":
                decoded.append(char)
                index += 1
                continue
            if index + 1 >= len(buffer):
                break  # Wait for the rest of the escape sequence.
            if buffer[index + 1] != "u":
                decoded.append(self.ESCAPES.get(buffer[index + 1], buffer[index + 1]))
                index += 2
                continue
            if index + 6 > len(buffer):
                break
            code = int(buffer[index + 2:index + 6], 16)
            if 0xD800 <= code < 0xDC00:  # A surrogate pair arrives as two escapes.
                if index + 12 > len(buffer):
                    break
                low = int(buffer[index + 8:index + 12], 16)
                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                index += 6
            decoded.append(chr(code))
            index += 6
        self.position = index
        if decoded:
            self.emit("".join(decoded))


def response_metrics(result):
    metrics = {}
    for source, label in (("load_duration", "load_seconds"), ("prompt_eval_duration", "prompt_seconds"),
                          ("eval_duration", "generation_seconds"), ("total_duration", "total_seconds")):
        value = result.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            metrics[label] = round(value / 1_000_000_000, 3)
    for source, label in (("prompt_eval_count", "prompt_tokens"), ("eval_count", "generated_tokens"),
                          ("prompt_eval_cached_count", "cached_tokens")):
        value = result.get(source)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            metrics[label] = value
    duration = result.get("eval_duration")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0 and "generated_tokens" in metrics:
        metrics["tokens_per_second"] = round(metrics["generated_tokens"] / (result["eval_duration"] / 1e9), 2)
    if metrics and "cached_tokens" not in metrics:
        metrics["cached_tokens"] = None
    return metrics


def stub(message):
    """One-line stand-in for an old bulky message. Tool results keep their header line."""
    content = message["content"]
    size = len(content.encode("utf-8"))
    if size <= STUB_MIN_BYTES:
        return message
    if message["role"] == "assistant":
        try:
            reply = json.loads(content)
        except json.JSONDecodeError:
            return message
        body = reply.get("content") if isinstance(reply, dict) else None
        if not isinstance(body, str) or len(body) <= 200:
            return message
        return {**message, "content": json.dumps({**reply, "content": f"[{len(body.encode()):,} bytes omitted]"})}
    header = content.split("\n", 1)[0][:200]
    if not header.startswith(("Tool result for", "Local reference excerpts")):
        return message
    return {**message, "content": f"{header}\n[Older result compacted: {size:,} bytes omitted. "
                                  "Read or search again if you need it.]"}


def compact_messages(messages, budget=12_000):
    """Keep a stable prefix until bounded history requires compaction.

    Preserve system instructions and the original task. First replace older tool results
    and file contents with one-line stubs (oldest first), keeping the conversation's shape.
    Only if that is not enough, drop the oldest messages and truncate the newest.
    Never mutate the graph's complete in-memory conversation.
    """
    def size(message):
        return len(message["content"].encode("utf-8")) + 64

    if sum(map(size, messages)) <= budget:
        return messages
    if len(messages) < 2:
        raise ValueError("Prompt exceeds budget; shorten the task or increase --prompt-budget")
    fixed = messages[:2]
    remaining = budget - sum(map(size, fixed)) - 160
    if remaining < 512:
        raise ValueError("Task and instructions exceed prompt budget; shorten the task or increase --prompt-budget")
    history = list(messages[2:])
    for index in range(len(history) - 1):  # The newest message stays intact.
        if sum(map(size, fixed + history)) <= budget:
            break
        history[index] = stub(history[index])
    if sum(map(size, fixed + history)) <= budget:
        return fixed + history
    recent = []
    for message in reversed(history):
        if size(message) <= remaining:
            recent.append(message)
            remaining -= size(message)
        elif not recent:
            marker = "\n[CONTEXT TRUNCATED: incomplete evidence; do not reconstruct a file from this excerpt.]"
            available = remaining - len(marker.encode()) - 64
            excerpt = message["content"].encode()[:max(0, available)].decode("utf-8", errors="ignore")
            recent.append({**message, "content": excerpt + marker})
            break
        else:
            break
    notice = {"role": "user", "content": "Earlier context omitted to fit budget; use tools to verify missing evidence."}
    return fixed + [notice] + list(reversed(recent))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Model server redirects are disabled")


class Ollama:
    # Waiting on a local model is normal: loading plus prompt processing alone can take minutes on a busy machine.
    def __init__(self, model=MODEL, endpoint="http://127.0.0.1:11434", timeout=600,
                 max_output_tokens=1024, prompt_budget=12_000):
        if not 256 <= max_output_tokens <= 8192:
            raise ValueError("--max-output-tokens must be between 256 and 8192")
        if not 8000 <= prompt_budget <= 64_000:
            raise ValueError("--prompt-budget must be between 8000 and 64000 bytes")
        self.max_output_tokens = max_output_tokens
        self.prompt_budget = prompt_budget
        self.last_metrics = {}
        self.last_raw = ""
        url = urllib.parse.urlsplit(endpoint)
        try:
            local = ipaddress.ip_address(url.hostname).is_loopback
            url.port  # Validate the optional port before making a request.
        except (ValueError, TypeError):
            local = False
        if (not local or url.scheme != "http" or url.username or url.password
                or url.path not in ("", "/") or url.query or url.fragment):
            raise ValueError("Endpoint must be an HTTP numeric loopback origin, e.g. http://127.0.0.1:11434")
        if not isinstance(model, str) or not MODEL_NAME.fullmatch(model) or cloud_name(model):
            raise ValueError(f"Not a local Ollama model name: {model!r}")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.endpoint + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ValueError("Model response exceeds 2 MB")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise ValueError("Model server returned a non-object response")
                return result
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ValueError(f"Local Ollama request failed: {exc}. Start Ollama and check the installed model.") from exc

    def stream_chat(self, payload, on_text=None):
        """POST a streaming chat, guard the finish answer against loops, and pass its text to on_text.

        Raising inside the read loop closes the connection, which stops Ollama generating.
        """
        req = urllib.request.Request(self.endpoint + "/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        guard = RepetitionGuard()

        def answer_text(text):
            guard.feed(text)
            if on_text is not None:
                on_text(text)

        decoder = FinishContent(answer_text)
        pieces, received, final = [], 0, {}
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                for raw in response:
                    received += len(raw)
                    if received > MAX_RESPONSE_BYTES:
                        raise ValueError("Model response exceeds 2 MB")
                    if not raw.strip():
                        continue
                    chunk = json.loads(raw)
                    if not isinstance(chunk, dict):
                        raise ValueError("Model server returned a non-object response")
                    if chunk.get("error"):
                        # Ollama itself stopped a looping reply; that is a bad reply, not a broken server.
                        if "repeat" in str(chunk["error"]).lower():
                            self.last_raw = "".join(pieces)
                            raise InvalidReply("the reply kept repeating the same text, so generation was stopped; "
                                               "reply briefly and state each point only once")
                        raise ValueError(f"Local Ollama error: {chunk['error']}")
                    piece = (chunk.get("message") or {}).get("content") or ""
                    if piece:
                        pieces.append(piece)
                        decoder.feed(piece)
                    if chunk.get("done"):
                        final = chunk
                        break
            self.last_raw = "".join(pieces)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ValueError(f"Local Ollama request failed: {exc}. Start Ollama and check the installed model.") from exc
        return {**final, "message": {"content": "".join(pieces)}}

    def check_model(self):
        models = self.request("/api/tags").get("models", [])
        names = {m.get("name") for m in models}
        if self.model not in names:
            raise ValueError(f"Model {self.model!r} is not installed. Run: ollama pull {self.model} while online.")
        info = self.request("/api/show", {"model": self.model})
        if info.get("remote_host") or info.get("remote_model"):
            raise ValueError("Remote-backed models are disabled")
        if "capabilities" in info and "completion" not in info["capabilities"]:
            raise ValueError(f"Model {self.model!r} cannot chat (it is an embedding model); choose another.")

    def local_models(self):
        """Installed models that run on this machine and can chat, the default first.

        Cloud-backed models and embedding-only models are left out.
        """
        found = []
        for item in self.request("/api/tags").get("models", []):
            name = item.get("name") if isinstance(item, dict) else None
            if (not isinstance(name, str) or cloud_name(name) or item.get("remote_host") or item.get("remote_model")
                    or ("capabilities" in item and "completion" not in item["capabilities"])):
                continue
            details = item.get("details") or {}
            found.append({"name": name, "parameters": details.get("parameter_size", ""), "family": details.get("family", ""),
                          "bytes": item.get("size", 0)})
        return sorted(found, key=lambda m: (m["name"] != MODEL, m["name"]))

    def use(self, name):
        """Switch to another installed local model; the current one stays if the new one fails its checks."""
        if not isinstance(name, str) or not MODEL_NAME.fullmatch(name) or cloud_name(name):
            raise ValueError(f"Not a local Ollama model name: {name!r}")
        previous, self.model = self.model, name
        try:
            self.check_model()
        except ValueError:
            self.model = previous
            raise

    def other_loaded_models(self):
        """Other models Ollama holds in memory: competing for memory makes it reload ours on every step."""
        try:
            running = self.request("/api/ps").get("models", [])
        except ValueError:
            return []
        return sorted({m.get("name") for m in running if isinstance(m, dict) and m.get("name") not in (None, self.model)})

    def chat(self, messages, schema, on_text=None):
        """Return one JSON action. Always streams from Ollama so a looping answer can be stopped early;
        with on_text, the finish answer is also shown as it is generated."""
        self.last_metrics = {}
        self.last_raw = ""
        payload = {"model": self.model, "messages": compact_messages(messages, self.prompt_budget),
                   "stream": True, "format": schema,
                   "keep_alive": "30m",
                   "options": {"temperature": 0, "num_predict": self.max_output_tokens,
                               "num_ctx": max(8192, self.prompt_budget // 2 + self.max_output_tokens)}}
        result = self.stream_chat(payload, on_text)
        self.last_metrics = response_metrics(result)
        if result.get("done_reason") == "length":
            raise InvalidReply("Model reached the response token limit; no partial action was executed. "
                               "Use edit with smaller SEARCH/REPLACE blocks or a shorter answer, "
                               "or increase --max-output-tokens.")
        try:
            return json.loads(result["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise InvalidReply("Model returned an invalid JSON response") from exc
