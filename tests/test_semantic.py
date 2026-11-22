"""Search by meaning: optional, local, mixed with key-word ranking, and never able to fail an answer."""
import os
import json
import stat
import tempfile
import unittest
from pathlib import Path

from offline_agent.knowledge import KnowledgeBase
from offline_agent.semantic import Semantic, choose_model
from offline_agent.workspace import Workspace

# A toy embedding: each axis is a topic, so texts about the same thing point the same way whatever words they use.
TOPICS = [("cancel", "penalty", "late", "last minute", "forfeit"), ("holiday", "closed", "shut"), ("price", "cost", "rate")]


class FakeEmbedder:
    def __init__(self, tags=None):
        self.calls, self.fail, self.companions = [], False, set()
        self.tags = tags if tags is not None else [{"name": "nomic-embed-text:latest", "capabilities": ["embedding"]}]

    def request(self, path, payload=None):
        if path == "/api/tags":
            return {"models": self.tags}
        if self.fail:
            raise ValueError("Local Ollama request failed")
        self.calls.append(payload["input"])
        return {"embeddings": [[float(sum(text.lower().count(word) for word in words)) for words in TOPICS] + [0.01]
                               for text in payload["input"]]}


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name, "shop")
        self.root.mkdir()
        (self.root / "refunds.py").write_text("def forfeit(booking):\n    return booking.deposit  # penalty kept\n")
        (self.root / "calendar.py").write_text("SHUT = ['2027-01-01']  # holiday list\n")
        (self.root / "words.py").write_text("def people():\n    return 'who do we charge'\n")
        self.workspace = Workspace(self.root)
        self.client = FakeEmbedder()
        self.cache = Path(self.temp.name, "index", "shop.vectors")

    def knowledge(self, **options):
        semantic = Semantic(self.client, "nomic-embed-text:latest", **options)
        base = KnowledgeBase(workspace=self.workspace, context_budget=4000, semantic=semantic)
        self.assertTrue(semantic.wait(10))
        return base, semantic

    def sources(self, base, query):
        return [row["source"] for row in json.loads(base.retrieve(query))["results"]]

    def test_meaning_finds_what_key_words_miss(self):
        query = "where do we charge people who cancel at the last minute?"
        plain = KnowledgeBase(workspace=self.workspace, context_budget=4000)
        self.assertNotEqual(self.sources(plain, query)[:1], ["repo/refunds.py"])
        base, _ = self.knowledge()
        self.assertEqual(self.sources(base, query)[0], "repo/refunds.py")
        self.assertTrue(json.loads(base.retrieve(query))["meaning_search"])
        self.assertFalse(json.loads(plain.retrieve(query))["meaning_search"])

    def test_documents_and_questions_get_the_models_prefixes(self):
        base, _ = self.knowledge()
        base.retrieve("closed days")
        self.assertTrue(all(text.startswith("search_document: ") for batch in self.client.calls[:-1] for text in batch))
        self.assertEqual(self.client.calls[-1], ["search_query: closed days"])

    def test_vectors_are_saved_privately_outside_the_repository_and_reused(self):
        before = sorted(path.name for path in self.root.iterdir())
        self.knowledge(cache=self.cache)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), before)
        if os.name != "nt":  # Windows keeps a user's files private through the profile folder, not file modes.
            self.assertEqual(stat.S_IMODE(self.cache.stat().st_mode), 0o600)
        self.assertNotIn(b"forfeit", self.cache.read_bytes())  # Numbers and hashes only, never code text.
        self.client.calls.clear()
        base, semantic = self.knowledge(cache=self.cache)
        self.assertEqual(self.client.calls, [])  # Nothing is embedded twice.
        (self.root / "calendar.py").write_text("SHUT = ['2027-12-25']  # holiday list\n")
        base.refresh()
        self.assertTrue(semantic.wait(10))
        self.assertEqual([len(batch) for batch in self.client.calls], [1])  # Only the changed passage.
        semantic.forget()
        self.assertFalse(self.cache.exists())

    def test_without_a_cache_nothing_is_written(self):
        self.knowledge()
        self.assertFalse(self.cache.parent.exists())

    def test_a_failing_embedding_model_falls_back_to_key_words(self):
        notes = []
        base, semantic = self.knowledge(output=notes.append)
        self.client.fail = True
        self.assertIn("repo/calendar.py", self.sources(base, "holiday list"))
        self.assertTrue(semantic.failed)
        self.assertEqual(len(notes), 1)
        self.assertFalse(json.loads(base.retrieve("holiday list"))["meaning_search"])

    def test_a_damaged_cache_is_ignored(self):
        self.cache.parent.mkdir(parents=True)
        self.cache.write_bytes(b"not a cache")
        base, semantic = self.knowledge(cache=self.cache)
        ready, total = semantic.coverage()
        self.assertTrue(ready == total >= 3)

    def test_choosing_the_embedding_model(self):
        tags = [{"name": "qwen2.5-coder:7b", "capabilities": ["completion"]},
                {"name": "embeddinggemma:latest", "capabilities": ["embedding"]},
                {"name": "nomic-embed-text:latest", "capabilities": ["embedding"]},
                {"name": "big-embed:cloud", "capabilities": ["embedding"]}]
        client = FakeEmbedder(tags)
        self.assertEqual(choose_model(client), "nomic-embed-text:latest")
        self.assertEqual(choose_model(client, "embeddinggemma"), "embeddinggemma:latest")
        self.assertIsNone(choose_model(FakeEmbedder(tags[:1])))
        self.assertIsNone(choose_model(object()))
        with self.assertRaisesRegex(ValueError, "not installed"):
            choose_model(client, "mxbai-embed-large")
        with self.assertRaisesRegex(ValueError, "chat model"):
            choose_model(client, "qwen2.5-coder:7b")
        with self.assertRaisesRegex(ValueError, "not installed"):
            choose_model(client, "big-embed:cloud")


if __name__ == "__main__":
    unittest.main()
