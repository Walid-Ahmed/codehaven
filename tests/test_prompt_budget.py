"""Keeping a conversation inside the prompt budget: what is compacted, what is kept, and what is refused."""
import json
import unittest
from unittest.mock import patch

from offline_agent.model import Ollama, compact_messages


class PromptBudgetTests(unittest.TestCase):
    def test_fitting_history_is_unchanged_for_prefix_reuse(self):
        messages = [{"role": "system", "content": "rules"},
                    {"role": "user", "content": "task"}]
        self.assertIs(compact_messages(messages), messages)

    def test_compaction_preserves_task_recent_result_and_unicode_budget(self):
        messages = [{"role": "system", "content": "rules"},
                    {"role": "user", "content": "original task"},
                    {"role": "user", "content": "old" * 9000},
                    {"role": "assistant", "content": "read file"},
                    {"role": "user", "content": "漢" * 9000}]
        original = json.dumps(messages)
        result = compact_messages(messages, 8000)
        self.assertEqual(result[:2], messages[:2])
        self.assertIn("TRUNCATED", result[-1]["content"])
        self.assertLessEqual(sum(len(m['content'].encode()) + 64 for m in result), 8000)
        self.assertEqual(json.dumps(messages), original)

    def test_oversized_task_is_rejected_instead_of_losing_instructions(self):
        with self.assertRaisesRegex(ValueError, "shorten"):
            compact_messages([{"role": "system", "content": "rules"},
                              {"role": "user", "content": "x" * 9000}], 8000)

    def test_response_cap_and_warm_model_options(self):
        client = Ollama()
        reply = {"action": "finish", "path": "", "message": "Done", "content": "Answer"}
        with patch.object(client, "stream_chat", return_value={"message": {"content": json.dumps(reply)}}) as request:
            self.assertEqual(client.chat([], {}), reply)
        payload = request.call_args.args[0]
        self.assertTrue(payload['stream'])  # Always streamed so a looping answer can be stopped early.
        self.assertEqual(payload['keep_alive'], '30m')
        self.assertEqual(payload['options']['num_predict'], 1024)
        self.assertEqual(payload['model'], 'qwen2.5-coder:7b')

    def test_capped_response_never_becomes_an_action_even_if_valid_json(self):
        client = Ollama()
        with patch.object(client, "stream_chat", return_value={"done_reason": "length", "message": {"content": '{}'}}):
            with self.assertRaisesRegex(ValueError, "no partial action"):
                client.chat([], {})

    def test_invalid_limits_rejected(self):
        for kwargs in ({"max_output_tokens": 0}, {"max_output_tokens": 8193},
                       {"prompt_budget": 7999}, {"prompt_budget": 64001}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Ollama(**kwargs)
