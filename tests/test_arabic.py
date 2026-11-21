"""Arabic questions: read for what they ask, matched in search across spellings, and answered in Arabic."""
import json
import tempfile
import unittest
from pathlib import Path

from offline_agent import arabic, changes, insights
from offline_agent.agent import REVIEW, Session, task_kind, trim_at_limit
from offline_agent.gui import upload_name
from offline_agent.knowledge import KnowledgeBase, tokens
from offline_agent.memory import terms
from offline_agent.workspace import Workspace


class ScriptedModel:
    prompt_budget = 12_000

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def chat(self, messages, _schema, on_text=None):
        self.prompts.append(messages)
        return self.replies.pop(0)


class ArabicTests(unittest.TestCase):
    def test_spelling_variants_become_one_word(self):
        self.assertEqual(arabic.normalise("أَيْنَ الدَّالَّة؟"), "اين الداله؟")
        self.assertEqual(arabic.normalise("إلى مكتبــة آخر ١٢"), "الي مكتبه اخر 12")
        self.assertEqual(tokens("الدَّالَّة"), tokens("الدالة"))
        self.assertEqual(tokens("الملفات"), tokens("ملف"))
        self.assertEqual(tokens("للمستخدمين"), tokens("مستخدم"))
        self.assertEqual(arabic.stem("من"), "من")  # Short words are left whole.

    def test_english_words_are_stemmed_as_before(self):
        self.assertEqual(tokens("applyDiscount restocking prices café"), ["apply", "discount", "restock", "pric", "café"])

    def test_the_language_of_a_task(self):
        for task in ("اشرح store.py", "ما وظيفة الدالة applyCoupon في checkout؟", "أين يتم حساب الضريبة؟",
                     "راجع shop/store.py", "اشرح الدالة `with_tax` من فضلك"):
            self.assertTrue(arabic.written_in(task), task)
        for task in ("Explain store.py", 'Where is the string "مرحبا بالعالم" printed?', "What does the مرحبا message mean here?",
                     "explain `تحية` in greet.py", ""):
            self.assertFalse(arabic.written_in(task), task)

    def test_what_an_arabic_task_asks_for(self):
        for task, kind in (("ما وظيفة الدالة with_tax؟", "question"), ("أين يتم حساب الضريبة", "question"),
                           ("اشرح الملف store.py", "question"), ("كيف يعمل المشروع؟", "question"),
                           ("هل هناك اختبارات", "question"), ("لماذا تفشل الدالة with_tax", "question"),
                           ("أصلح الخطأ في store.py", "change"), ("أضف دالة لحساب الخصم", "change"),
                           ("من فضلك غيّر اسم الدالة add", "change"), ("هل يمكنك إصلاح الخطأ في add؟", "change"),
                           ("لو سمحت احذف الدالة _log", "change"),
                           ("اشرح الملف store.py بدون تعديل أي شيء", "question"), ("اشرح الدالة ولا تغير شيئا", "question"),
                           ("ما الدوال غير المستخدمة؟", "question"),  # غير here means "not", so it is no request to change.
                           ("كيف أضيف دالة جديدة وأصلح الخطأ", "other"), ("store.py", "other")):
            self.assertEqual(task_kind(task), kind, task)

    def test_english_tasks_are_read_as_before(self):
        for task, kind in (("What does with_tax do?", "question"), ("Fix the bug in store.py", "change"),
                           ("Can you rename add?", "change"), ("Explain store.py without changing anything", "question")):
            self.assertEqual(task_kind(task), kind, task)

    def test_arabic_requests_for_changes_reviews_gaps_and_tours(self):
        for task, wanted in (("اشرح تغييراتي", ("uncommitted", None)), ("ماذا غيّرت؟", ("uncommitted", None)),
                             ("اشرح التغييرات غير المودعة", ("uncommitted", None)),
                             ("ماذا تغير منذ آخر commit؟", ("uncommitted", None)),
                             ("ماذا تغير في آخر commit؟", ("commit", "HEAD")), ("اشرح الكوميت الأخير", ("commit", "HEAD")),
                             ("اشرح التغييرات المرحلة", ("staged", None)), ("اشرح الكوميت abc1234", None),
                             ("اشرح commit abc1234", ("commit", "abc1234")), ("اشرح الملف store.py", None)):
            self.assertEqual(changes.requested(task), wanted, task)
        for task, wanted in (("أي الدوال بدون اختبارات؟", "gaps"), ("ما الدوال غير المختبرة؟", "gaps"),
                             ("ما هي فجوات الاختبارات؟", "gaps"), ("أعطني جولة في المشروع", "tour"),
                             ("من أين أبدأ؟", "tour"), ("أي الملفات يجب أن أقرأ أولا؟", "tour"),
                             ("ما هي الاختبارات الموجودة؟", None), ("اشرح store.py", None)):
            self.assertEqual(insights.requested(task), wanted, task)
        for task in ("راجع store.py", "من فضلك راجع store.py", "مراجعة store.py", "ابحث عن الأخطاء في store.py"):
            self.assertTrue(arabic.REVIEW.search(arabic.normalise(task)), task)
        for task in ("اشرح store.py", "ما الذي يراجعه هذا الملف؟"):
            self.assertFalse(arabic.REVIEW.search(arabic.normalise(task)) or REVIEW.search(task), task)

    def test_common_arabic_words_do_not_decide_which_note_is_recalled(self):
        self.assertEqual(terms("ما هي وظيفة الدالة with_tax في المشروع؟"), {"وظيف", "tax"})

    def test_an_answer_cut_at_its_limit_ends_at_an_arabic_question_mark(self):
        text = "هل تعمل الدالة؟ نعم تعمل. ثم ماذا؟ بقية الجملة المقطو"
        self.assertEqual(trim_at_limit(text, len(text)), ("هل تعمل الدالة؟ نعم تعمل. ثم ماذا؟", True))

    def test_uploaded_files_keep_arabic_names(self):
        self.assertEqual(upload_name("ملف الأسعار.py"), "ملف_الأسعار.py")
        self.assertEqual(upload_name("café.py"), "caf_.py")


class ArabicProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name, "shop")
        for name, text in {
            "pricing.py": "# حساب الضَّريبة على السعر\ndef with_tax(price):\n    return price * 1.2\n",
            "store.py": "class Store:\n    def add(self, item):\n        return item\n",
            "docs/notes.md": "# ملاحظات\n\nالمخزون يتم تحديثه بعد كل عملية بيع.\n",
        }.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.workspace = Workspace(self.root)

    def test_an_arabic_question_finds_arabic_comments_and_documents(self):
        knowledge = KnowledgeBase(workspace=self.workspace)
        found = json.loads(knowledge.retrieve("أين تُحسب الضريبة؟"))["results"]
        self.assertTrue(found and found[0]["source"].endswith("pricing.py"), found)
        found = json.loads(knowledge.retrieve("متى يتغير المخزون"))["results"]
        self.assertTrue(found and found[0]["source"].endswith("docs/notes.md"), found)

    def ask(self, task):
        model = ScriptedModel([{"action": "finish", "path": "", "message": "", "content": "تضيف الضريبة إلى السعر.",
                                "points": [{"path": "pricing.py", "symbol": "with_tax", "text": "تضرب السعر في 1.2"}]}])
        lines = []
        session = Session(model, self.workspace, output=lines.append)
        session.ask(task)
        return model.prompts[0], session.last_state

    def test_an_arabic_question_is_answered_in_arabic_with_real_line_citations(self):
        prompt, state = self.ask("اشرح الدالة with_tax في pricing.py")
        self.assertTrue(prompt[-1]["content"].endswith(arabic.ANSWER_IN_ARABIC))
        self.assertEqual(state["kind"], "question")
        self.assertEqual(state["reply"]["content"], "تضيف الضريبة إلى السعر.\n- pricing.py:2-3 `with_tax`: تضرب السعر في 1.2")
        self.assertEqual(state["exit_code"], 0, state.get("issues"))

    def test_an_english_question_gets_the_same_prompt_as_before(self):
        prompt, _ = self.ask("Explain with_tax in pricing.py")
        self.assertFalse(any("Arabic" in message["content"] for message in prompt[1:]))


if __name__ == "__main__":
    unittest.main()
