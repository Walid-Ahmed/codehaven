"""Ollama transport: numeric loopback only, no proxies or redirects."""
import contextlib
import http.client
import ipaddress
import json
import re
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request

MODEL = "qwen2.5-coder:7b"  # The default, and the model the evals were tuned and measured on.
# The characters an Ollama model name may use; anything else is refused before it reaches a request.
MODEL_NAME = re.compile(r"[\w.:/@-]{1,200}")
# The most bytes accepted in one model reply, so a misbehaving server cannot fill memory.
MAX_RESPONSE_BYTES = 2_000_000
# Messages this small are left as they are when compacting: a stub would save almost nothing.
STUB_MIN_BYTES = 400
PROMPT_BUDGET = 12_000  # Default conversation budget in UTF-8 bytes; --prompt-budget changes it.


def cloud_name(name):
    """Ollama names cloud-backed models with a cloud tag; CodeHaven never sends code to them."""
    return name.endswith(("-cloud", ":cloud"))


def read_limit(model):
    """The most bytes of one file to put in a prompt: half the model's budget, so the rest of the conversation fits."""
    return max(4000, getattr(model, "prompt_budget", PROMPT_BUDGET) // 2)


class InvalidReply(ValueError):
    """The model answered, but not with a usable action; the agent may ask it to correct itself."""


class Stopped(Exception):
    """The user stopped the answer. Not a ValueError, so no helper mistakes it for a bad reply and carries on."""

    def __init__(self):
        super().__init__("Stopped.")


class RepeatedAnswer(InvalidReply):
    """A finish answer began looping and generation was stopped. answer holds the text before the loop."""

    def __init__(self, answer):
        """Keep the usable part of the answer, and word the error as a correction the model can act on."""
        super().__init__("the final answer kept repeating the same text, so generation was stopped; "
                         "answer briefly and state each point only once")
        self.answer = answer


class RepetitionGuard:
    """Stop a final answer that keeps repeating the same phrase, as small models can at temperature 0.

    Checked on every streamed piece, so the window slides across a loop's shared phrase even when
    the surrounding words drift. A global repeat penalty is avoided because answers quote code, which must be copied exactly.
    """
    # A loop is the latest WINDOW characters having appeared TIMES times; that tail must hold at least
    # DISTINCT different characters to count.
    WINDOW, TIMES, DISTINCT = 60, 3, 8

    def __init__(self):
        """Start with no answer text seen."""
        self.text = ""

    def feed(self, piece):
        """Add one streamed piece of the answer; raise RepeatedAnswer once it is looping."""
        self.text += piece
        # Too short to hold TIMES copies of a window.
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
        """position is the buffer index of the next undecoded character, or None until the content string opens."""
        self.emit = emit
        self.buffer = ""
        self.position = None
        self.done = False

    def feed(self, text):
        """Add raw streamed JSON text and emit whatever more of the finish content can now be decoded."""
        self.buffer += text
        if self.done:
            return
        # Emit nothing until the reply is known to be a finish action and its content string has opened.
        if self.position is None:
            content = re.search(r'"content"\s*:\s*"', self.buffer)
            if not content or not re.search(r'"action"\s*:\s*"finish"', self.buffer):
                return
            self.position = content.end()
        decoded, index, buffer = [], self.position, self.buffer
        while index < len(buffer):
            char = buffer[index]
            # An unescaped quote closes the content string.
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
            # A \uXXXX escape: wait until all four hex digits have arrived.
            if index + 6 > len(buffer):
                break
            code = int(buffer[index + 2:index + 6], 16)
            if 0xD800 <= code < 0xDC00:  # A surrogate pair arrives as two escapes.
                if index + 12 > len(buffer):
                    break
                low = int(buffer[index + 8:index + 12], 16)
                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                # Step over the high half here; the shared code below steps over the low half.
                index += 6
            decoded.append(chr(code))
            index += 6
        self.position = index
        if decoded:
            self.emit("".join(decoded))


def response_metrics(result):
    """Turn the counters in Ollama's final chunk into seconds, token counts, and tokens per second.

    Missing or malformed counters are left out rather than guessed. bool is excluded explicitly because
    it is a kind of int in Python.
    """
    metrics = {}
    for source, label in (("load_duration", "load_seconds"), ("prompt_eval_duration", "prompt_seconds"),
                          ("eval_duration", "generation_seconds"), ("total_duration", "total_seconds")):
        value = result.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            # Ollama reports durations in nanoseconds.
            metrics[label] = round(value / 1_000_000_000, 3)
    for source, label in (("prompt_eval_count", "prompt_tokens"), ("eval_count", "generated_tokens"),
                          ("prompt_eval_cached_count", "cached_tokens")):
        value = result.get(source)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            metrics[label] = value
    duration = result.get("eval_duration")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0 and "generated_tokens" in metrics:
        metrics["tokens_per_second"] = round(metrics["generated_tokens"] / (duration / 1e9), 2)
    # Not every Ollama version reports cached tokens; None says unknown, which is different from zero.
    if metrics and "cached_tokens" not in metrics:
        metrics["cached_tokens"] = None
    return metrics


def stub(message):
    """One-line stand-in for an old bulky message. Tool results keep their header line."""
    content = message["content"]
    size = len(content.encode("utf-8"))
    if size <= STUB_MIN_BYTES:
        return message
    # An old assistant action keeps its JSON shape; only a long content field is replaced.
    if message["role"] == "assistant":
        try:
            reply = json.loads(content)
        except json.JSONDecodeError:
            return message
        body = reply.get("content") if isinstance(reply, dict) else None
        if not isinstance(body, str) or len(body) <= 200:
            return message
        return {**message, "content": json.dumps({**reply, "content": f"[{len(body.encode()):,} bytes omitted]"})}
    # Other messages are compacted only when they are tool results or reference excerpts, never user text.
    header = content.split("\n", 1)[0][:200]
    if not header.startswith(("Tool result for", "Local reference excerpts")):
        return message
    return {**message, "content": f"{header}\n[Older result compacted: {size:,} bytes omitted. "
                                  "Read or search again if you need it.]"}


def compact_messages(messages, budget=PROMPT_BUDGET):
    """Keep a stable prefix until bounded history requires compaction.

    Preserve system instructions and the original task. First replace older tool results
    and file contents with one-line stubs (oldest first), keeping the conversation's shape.
    Only if that is not enough, drop the oldest messages and truncate the newest.
    Never mutate the graph's complete in-memory conversation.
    """
    def size(message):
        """Bytes a message costs in the prompt, allowing 64 for its role and framing."""
        return len(message["content"].encode("utf-8")) + 64

    if sum(map(size, messages)) <= budget:
        return messages
    if len(messages) < 2:
        raise ValueError("Prompt exceeds budget; shorten the task or increase --prompt-budget")
    # The system instructions and the original task are never compacted.
    # 160 bytes are held back for the "Earlier context omitted" notice added at the end.
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
    # Stubs were not enough: keep the newest messages that fit, and cut the newest one if even it does not fit.
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
    """Refuse every HTTP redirect, so a reply can never send the prompt (and the code in it) to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Called by urllib for any 3xx reply; raising here stops the redirect from being followed."""
        raise ValueError("Model server redirects are disabled")


class ChatHandler(urllib.request.HTTPHandler):
    """Opens connections like the standard handler, and lets the client keep hold of a chat's socket.

    urllib hides the socket while it waits for the first byte of a reply, which with a local model can be
    minutes of loading and prompt processing. Keeping the socket is what lets Ollama.stop() end that wait.
    """

    def __init__(self, client):
        super().__init__()
        self.client = client

    def http_open(self, req):
        """Open req on a connection that reports its socket to the client as soon as it is connected."""
        client = self.client

        class Connection(http.client.HTTPConnection):
            """A plain HTTP connection that tells the client about its socket."""

            def connect(self):
                """Connect, then hand the socket over; this raises Stopped if the answer was already stopped."""
                super().connect()
                client.connected(self.sock)

        return self.do_open(Connection, req)


class Ollama:
    """Client for one local Ollama server. Only plain HTTP on a numeric loopback address is accepted."""

    # Waiting on a local model is normal: loading plus prompt processing alone can take minutes on a busy machine.
    def __init__(self, model=MODEL, endpoint="http://127.0.0.1:11434", timeout=600,
                 max_output_tokens=1024, prompt_budget=PROMPT_BUDGET):
        """Validate the limits, the endpoint, and the model name. Nothing is sent until a request is made."""
        if not 256 <= max_output_tokens <= 8192:
            raise ValueError("--max-output-tokens must be between 256 and 8192")
        if not 8000 <= prompt_budget <= 64_000:
            raise ValueError("--prompt-budget must be between 8000 and 64000 bytes")
        self.max_output_tokens = max_output_tokens
        self.prompt_budget = prompt_budget
        self.last_metrics = {}
        self.last_raw = ""
        self.companions = set()  # Other models CodeHaven itself uses, such as the embedding model.
        # The host must be a numeric loopback IP: a name such as localhost could be pointed somewhere else.
        url = urllib.parse.urlsplit(endpoint)
        try:
            local = ipaddress.ip_address(url.hostname).is_loopback
            url.port  # Validate the optional port before making a request.
        except (ValueError, TypeError):
            local = False
        # Credentials, a path, a query, or a fragment are refused too, so the endpoint is exactly an origin.
        if (not local or url.scheme != "http" or url.username or url.password
                or url.path not in ("", "/") or url.query or url.fragment):
            raise ValueError("Endpoint must be an HTTP numeric loopback origin, e.g. http://127.0.0.1:11434")
        if not isinstance(model, str) or not MODEL_NAME.fullmatch(model) or cloud_name(model):
            raise ValueError(f"Not a local Ollama model name: {model!r}")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        # An empty ProxyHandler ignores the proxy environment variables, and NoRedirect refuses redirects,
        # so every request goes straight to the loopback address and nowhere else.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(), ChatHandler(self))
        # Stopping an answer: the flag stop() sets, the thread a chat is running on, and that chat's socket.
        self.stopping = threading.Event()
        self.chat_thread = self.chat_socket = None

    def connected(self, sock):
        """Called for every new connection: keep a chat's socket, so stop() can end a wait on it."""
        if threading.get_ident() != self.chat_thread:
            return  # Another request, such as an embedding, which stop() leaves alone.
        self.chat_socket = sock
        # stop() sets the flag before it looks for the socket, so one of the two always notices the other.
        if self.stopping.is_set():
            raise Stopped()

    def stop(self):
        """Stop the chat in progress, and refuse later ones until allow() is called. Safe from any thread.

        Closing the connection is what stops Ollama generating, and it ends a wait for a model that is still
        loading or reading its prompt, which a flag checked between pieces of the reply would not.
        """
        self.stopping.set()
        sock = self.chat_socket
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)

    def allow(self):
        """Let chats run again after a stop(); called when a new request begins."""
        self.stopping.clear()

    def request(self, path, payload=None):
        """Send one request (GET without a payload, POST with one) and return the JSON object in the reply."""
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.endpoint + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                # Reading one byte past the cap tells an oversized reply from one exactly at the cap.
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

        Raising inside the read loop closes the connection, which stops Ollama generating. stop(), called from
        another thread, ends this call with Stopped whether it is waiting for the reply or reading it.
        """
        if self.stopping.is_set():
            raise Stopped()
        req = urllib.request.Request(self.endpoint + "/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        guard = RepetitionGuard()

        def answer_text(text):
            """Check each decoded piece of the finish answer for looping, then show it."""
            guard.feed(text)
            if on_text is not None:
                on_text(text)

        decoder = FinishContent(answer_text)
        pieces, received, final = [], 0, {}
        self.chat_thread = threading.get_ident()
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                # Ollama streams one JSON object per line.
                for raw in response:
                    if self.stopping.is_set():
                        raise Stopped()
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
        except Stopped:
            raise
        except Exception as exc:
            # A connection closed by stop() fails in whatever way the read happened to be waiting.
            if self.stopping.is_set():
                raise Stopped() from None
            if isinstance(exc, (urllib.error.URLError, TimeoutError)):
                raise ValueError(f"Local Ollama request failed: {exc}. Start Ollama and check the installed model.") from exc
            raise
        finally:
            self.chat_thread = self.chat_socket = None
        if self.stopping.is_set():
            raise Stopped()  # The closed connection read as a reply that simply ended early.
        return {**final, "message": {"content": "".join(pieces)}}

    def check_model(self):
        """Raise ValueError unless the chosen model is installed, runs on this machine, and can chat."""
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
        return sorted({m.get("name") for m in running if isinstance(m, dict) and m.get("name") not in {None, self.model, *self.companions}})

    def chat(self, messages, schema, on_text=None):
        """Return one JSON action. Always streams from Ollama so a looping answer can be stopped early;
        with on_text, the finish answer is also shown as it is generated."""
        self.last_metrics = {}
        self.last_raw = ""
        # Temperature 0 keeps actions repeatable, and keep_alive holds the model in memory between steps.
        # num_ctx is in tokens while the budget is in bytes: half the budget plus the reply, and never below 8192.
        payload = {"model": self.model, "messages": compact_messages(messages, self.prompt_budget),
                   "stream": True, "format": schema,
                   "keep_alive": "30m",
                   "options": {"temperature": 0, "num_predict": self.max_output_tokens,
                               "num_ctx": max(8192, self.prompt_budget // 2 + self.max_output_tokens)}}
        result = self.stream_chat(payload, on_text)
        self.last_metrics = response_metrics(result)
        if result.get("done_reason") == "length":
            raise InvalidReply("Model reached the response token limit; no partial action was executed. "
                               "Give a shorter answer, or increase --max-output-tokens.")
        try:
            return json.loads(result["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise InvalidReply("Model returned an invalid JSON response") from exc
