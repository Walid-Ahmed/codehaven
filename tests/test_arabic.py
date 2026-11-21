"""Arabic questions: read for what they ask, matched in search across spellings, and answered in Arabic."""
import json
import re
import tempfile
import unittest
from pathlib import Path

from offline_agent import arabic, changes, insights, study
from offline_agent.agent import REVIEW, Session, task_kind, trim_at_limit
from offline_agent.gui import PAGE, upload_name
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

    def test_a_review_asked_for_in_arabic_is_reported_in_arabic(self):
        verdict = {"promise": "p", "does": "d", "edge": "e", "verdict": "bug", "finding": "تُرجع العنصر دون حفظه."}
        model = ScriptedModel([verdict])
        session = Session(model, self.workspace, output=lambda _line: None)
        session.ask("راجع store.py")
        self.assertTrue(model.prompts[0][-1]["content"].endswith(arabic.WRITE_IN_ARABIC))
        self.assertEqual(session.last_state["final_text"].splitlines(), [
            "تمت مراجعة دالة واحدة من store.py. وُجد: مشكلة محتملة واحدة. كل جزء فحصه النموذج المحلي وحده، لذلك اعتبر هذه "
            "النتائج خيوطًا للتحقق منها لا دليلًا قاطعًا. CodeHaven للقراءة فقط: لم يتغير أي شيء.",
            "- store.py:2-3 `Store.add` خطأ: تُرجع العنصر دون حفظه."])
        model = ScriptedModel([verdict])
        Session(model, self.workspace, output=lambda _line: None).ask("Review store.py")
        self.assertNotIn("Arabic", model.prompts[0][-1]["content"])

    def test_test_gaps_and_the_tour_asked_for_in_arabic_are_reported_in_arabic(self):
        (self.root / "main.py").write_text("from pricing import with_tax\nfrom store import Store\n", encoding="utf-8")
        (self.root / "tests").mkdir()
        (self.root / "tests/test_pricing.py").write_text("from pricing import with_tax\n", encoding="utf-8")
        knowledge = KnowledgeBase(workspace=self.workspace)
        gaps = insights.test_gaps(self.workspace, knowledge, in_arabic=True)["text"].splitlines()
        self.assertTrue(gaps[0].startswith("1 من 2 من الدوال لم يُذكر اسمها في أي من ملفات الاختبار (ملف واحد)."), gaps[0])
        self.assertEqual(gaps[1:], ["- store.py:2-3 `Store.add`"])
        self.assertTrue(insights.test_gaps(self.workspace, knowledge)["text"].startswith("1 of 2 functions and methods"))
        model = ScriptedModel([{"sentence": "يبدأ المتجر."}, {"sentence": "يحسب الضريبة."}, {"sentence": "يخزن العناصر."}])
        tour = insights.tour(model, self.workspace, knowledge, lambda _line: None, in_arabic=True)["text"].splitlines()
        self.assertTrue(tour[0].startswith("ترتيب لقراءة هذا المشروع: 3 من 4 من ملفات الكود"), tour[0])
        self.assertEqual(tour[1], "1. main.py:1-2 يبدأ المتجر. لماذا الآن: نقطة الدخول: يصل إلى أكبر جزء من المشروع عبر ما يستورده، ولا يستورده أي ملف.")
        self.assertEqual(tour[2], "2. pricing.py:1-3 يحسب الضريبة. لماذا الآن: يستورده main.py.")
        self.assertTrue(all(prompt[-1]["content"].endswith(arabic.WRITE_IN_ARABIC) for prompt in model.prompts))

    def test_flashcards_made_from_the_arabic_page_are_written_in_arabic(self):
        cards = {"cards": [{"question": "ماذا تفعل with_tax؟", "answer": "تضيف الضريبة.", "symbol": "with_tax"}]}
        for in_arabic in (True, False):
            model = ScriptedModel([cards])
            events = list(study.flashcards(model, self.workspace, ["pricing.py"], 3, in_arabic=in_arabic))
            self.assertEqual(events[1]["cards"][0]["start"], 2)
            self.assertEqual(model.prompts[0][-1]["content"].endswith(arabic.WRITE_IN_ARABIC), in_arabic)

    def test_counts_take_the_arabic_form_for_their_number(self):
        self.assertEqual([arabic.counted(number, *arabic.FILES) for number in (1, 2, 3, 10, 11, 100, 103)],
                         ["ملف واحد", "ملفان", "3 ملفات", "10 ملفات", "11 ملفًا", "100 ملفًا", "103 ملفات"])


class ArabicPageTests(unittest.TestCase):
    """The page is one file in two languages: its English text is the key each Arabic sentence is looked up by."""
    def setUp(self):
        self.page = PAGE.read_text(encoding="utf-8")
        table = self.page.split("const ARABIC = {", 1)[1].split("};", 1)[0]
        self.arabic = dict(json.loads("[" + line.strip().rstrip(",").replace('": "', '", "', 1) + "]")
                           for line in table.splitlines() if line.strip() and not line.strip().startswith("//"))
        nouns = self.page.split("const ARABIC_NOUNS = {", 1)[1].split("};", 1)[0]
        self.nouns = {json.loads(line.split(": [", 1)[0].strip()) for line in nouns.splitlines() if line.strip()}
        self.script = self.page.split("};", 2)[2]  # The code after both tables.

    def test_every_sentence_the_page_writes_has_an_arabic_one(self):
        asked = set(re.findall(r'\bt\((?:[^"`()]*\?\s*)?"((?:[^"\\]|\\.)*)"', self.script))
        asked |= {text for group in re.findall(r'\bt\(([^()]*?\?[^()]*?)(?:, \{|\))', self.script)  # t(this ? "One" : "Other")
                  for text in re.findall(r'"((?:[^"\\]|\\.)*)"', group)}
        asked = {json.loads(f'"{text}"') for text in asked}
        self.assertGreater(len(asked), 100)
        self.assertEqual(sorted(asked - set(self.arabic)), [])
        self.assertEqual(set(re.findall(r'\bcount\([^"\n]*?, "([a-z ]+)"\)', self.script)) - self.nouns, set())

    def test_the_page_text_written_in_html_has_an_arabic_sentence(self):
        body = self.page.split("<body>", 1)[1].split("<script", 1)[0]
        texts = {text.strip() for text in re.findall(r">([^<>]+)<", body)}
        texts |= set(re.findall(r'(?:title|placeholder|aria-label|data-drop)="([^"]+)"', body))
        ignored = {"CodeHaven", "☰", ".", "↑", "/Users/you/dev/project", "github.com/owner/repository", "العربية", "اعرض هذه الصفحة بالعربية"}
        missing = sorted(text for text in texts if text and text not in ignored and text not in self.arabic)
        self.assertEqual(missing, [])

    def test_arabic_sentences_keep_the_places_their_english_ones_fill(self):
        for english, translated in self.arabic.items():
            self.assertEqual(sorted(re.findall(r"\{\w+\}", english)), sorted(re.findall(r"\{\w+\}", translated)), english)
            self.assertTrue(re.search(r"[\u0621-\u064A]", translated), english)

    def test_arabic_suggestions_reach_the_same_helpers_as_the_english_ones(self):
        self.assertEqual(insights.requested("أعطني جولة في المشروع: أي الملفات يجب أن أقرأ أولا؟"), "tour")
        self.assertEqual(insights.requested("أي الدوال بدون اختبارات؟"), "gaps")
        self.assertIsNone(insights.requested("ما هي الاختبارات الموجودة وماذا تفحص؟"))
        self.assertEqual(changes.requested("اشرح تغييراتي"), ("uncommitted", None))
        self.assertEqual(changes.requested("ماذا تغير في آخر commit؟"), ("commit", "HEAD"))
        self.assertTrue(arabic.REVIEW.search(arabic.normalise("راجع store.py")))
        for question in ("اشرح هذا المشروع", "اشرح README.md", "ما هي الوحدات الرئيسية وما وظيفة كل منها؟"):
            self.assertTrue(arabic.written_in(question), question)
            self.assertIn(f'"{question}"', self.page)


if __name__ == "__main__":
    unittest.main()
