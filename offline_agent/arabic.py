"""Arabic questions: telling that a task is in Arabic, reading what it asks for, and matching its words in search.

Everything here works on text that normalise() has been through, so one spelling stands for all the ways a
word is typed: with or without short vowels, with any form of alef, and with a final ى or ة. The patterns
below are written in that spelling (اين, not أين).

Code is still written in Latin letters, so an Arabic question finds code through the names it mentions, through
search by meaning, and through Arabic comments, strings, and documentation in the project.
"""
import re

MARKS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # Short vowels and other marks, and the stretching dash.
FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه",
                      **{chr(0x0660 + digit): str(digit) for digit in range(10)},   # ٠-٩
                      **{chr(0x06F0 + digit): str(digit) for digit in range(10)}})  # ۰-۹
WORD = re.compile(r"[ء-ي]+")
LATIN_WORD = re.compile(r"[A-Za-z]+")
# Quoted text and code are what a question is about, not the language it is asked in.
QUOTED = re.compile(r"`[^`\n]*`|\"[^\"\n]*\"|'[^'\n]*'|«[^»\n]*»|\S*[/\\.]\S*|\S*[_\d]\S*")

PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
SUFFIXES = ("ها", "ان", "ات", "ون", "ين", "يه", "ه", "ي")

# "The text of every point" made the 7B model paste the code into each point; "what it does" keeps it a description.
ANSWER_IN_ARABIC = ("The user wrote in Arabic, so answer in Arabic: write content in Arabic, and in each point say in one "
                    "short Arabic sentence what that part does. Keep file paths, names from the code, and path:line "
                    "citations exactly as written, in Latin letters and Western digits; never translate a name from the code. "
                    "Do not repeat a sentence.")
WRITE_IN_ARABIC = "Write every sentence you return in Arabic, keeping names from the code exactly as they are written."

_POLITE = r"(?:من فضلك|لو سمحت|رجاء|فضلا|ممكن|هل يمكنك|هل يمكن|هل تستطيع|هل بامكانك|يا ريت)[\s،,]*(?:ان\s+)?"
POLITE = re.compile(rf"\W*{_POLITE}")
QUESTION = re.compile(r"\W*(?:ما|ماذا|ماهو|ماهي|من|اين|متي|لماذا|كيف|هل|كم|اي|اشرح|وضح|صف|لخص|اعرض|اظهر|اذكر|عدد|ابحث|جد|اوجد"
                      r"|حدد|افحص|راجع|اخبرني|قل لي|عرفني|ليه|ازاي|فين|ايه|وين|شو|ايش|وش)\b")
# Commands and the nouns made from them ("أصلح الخطأ", "هل يمكنك إصلاح الخطأ"). غير also means "not", so it
# counts only where a command stands: at the start. A command may carry "and" (وأصلح).
EDIT = re.compile(r"\b[وف]?(?:اصلح|صلح|صحح|عدل|حدث|اضف|احذف|ازل|امسح|انشئ|اكتب|نفذ|استبدل|اجعل|حول|انقل"
                  r"|اصلاح|تصحيح|تغيير|تعديل|تحديث|اضافه|حذف|ازاله|انشاء|كتابه|تنفيذ|استبدال|تحويل|نقل|اعاده)\b|^\W*غير\b")
NEGATED_EDIT = re.compile(r"\b(?:لا\s+ت|(?:بدون|دون|من غير)\s+(?:ال)?)\w+")
REVIEW = re.compile(rf"^\W*(?:{_POLITE})?(?:راجع|مراجعه|افحص|فحص|دقق|تدقيق|انقد)\b"
                    r"|\b(?:ابحث عن|جد|اوجد|ايجاد|اكتشف)\s+(?:اي\s+)?(?:ال)?(?:اخطاء|مشاكل|عيوب|ثغرات)\s+(?:في|ب)")

_LAST = r"(?:اخر|احدث|الاخير|الاخيره|السابق|السابقه)"
_COMMIT = r"(?:commit|كوميت|ايداع|الكوميت|الايداع|التزام|الالتزام)"
SINCE = re.compile(rf"\bمنذ\s+(?:{_LAST}\s+{_COMMIT}|{_COMMIT}\s+{_LAST})")
STAGED = re.compile(r"\b(?:ال)?(?:تغييرات|تعديلات)\s+ال(?:مرحله|مجهزه|مضافه)\b|\bstaged\b", re.I)
UNCOMMITTED = re.compile(r"\bغير\s+(?:ال)?(?:مودعه|مثبته|ملتزمه|محفوظه)\b|\bلم\s+(?:يتم\s+)?(?:ايداع|تثبيت|حفظ)"
                         r"|\b(?:تغييراتي|تعديلاتي)\b|\b(?:ال)?(?:تغييرات|تعديلات)\s+(?:الحاليه|المحليه|الاخيره)\b"
                         r"|\bماذا\s+(?:غيرت|عدلت)\b|\bما\s+الذي\s+(?:غيرته|عدلته|غيرت|عدلت)\b|\bgit\s+diff\b", re.I)
LAST_COMMIT = re.compile(rf"\b(?:{_LAST}\s+{_COMMIT}|{_COMMIT}\s+{_LAST})\b|\bgit\s+show\b", re.I)

GAPS = re.compile(r"\b(?:بدون|دون|بلا|ليس لها|ليس له|لا تملك|ينقصها|تنقصها)\s+(?:اي\s+)?(?:ال)?اختبار(?:ات)?\b"
                  r"|\bغير\s+(?:ال)?(?:مختبره|مختبر|مغطاه|مغطي)\b|\bلم\s+(?:يتم\s+)?(?:اختبار|تختبر|يختبر)"
                  r"|\b(?:فجوات|ثغرات|نواقص)\s+(?:ال)?اختبار(?:ات)?\b|\bتغطيه\s+(?:ال)?اختبار(?:ات)?\b")
TOUR = re.compile(r"\bجوله\s+(?:في|داخل|علي|حول)\b|\bجوله\s+(?:ال)?(?:مشروع|كود|مستودع)\b"
                  r"|\bمن\s+اين\s+(?:ابدا|ابدء|نبدا|يجب ان ابدا)\b|\b(?:ماذا|ما الذي)\s+(?:يجب ان\s+)?اقرا\s+اولا\b"
                  r"|\b(?:اي|ما)\s+(?:ال)?ملفات\s+(?:التي\s+)?(?:يجب ان\s+)?اقرا(?:ها)?\s+اولا\b|\bترتيب\s+(?:ال)?قراءه\b")

# Words that say nothing about which code a question concerns, as stem() returns them.
_COMMON = """ما ماذا من اين متي لماذا كيف هل كم اي في علي الي عن مع هذا هذه ذلك تلك هو هي التي الذي ان او و ثم لكن كل بعض
اشرح وضح صف اعرض اذكر اخبرني لي كود ملف داله دوال صنف مشروع مستودع يعمل تعمل يقوم تقوم"""


def normalise(text):
    """One spelling for each word: no short vowels or stretching, one alef, ي for ى, ه for ة, and Western digits."""
    return MARKS.sub("", text).translate(FOLD)


def stem(word):
    """Drop "the" and the common endings from a normalised word, so الدوال finds دوال and ملفات finds ملف."""
    for prefix in PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= 3:
            word = word[len(prefix):]
            break
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


COMMON = frozenset(stem(word) for word in normalise(_COMMON).split())


def written_in(task):
    """Whether a task is asked in Arabic. Quoted text, paths, and code names do not count either way."""
    plain = QUOTED.sub(" ", task)
    arabic, latin = len(WORD.findall(plain)), len(LATIN_WORD.findall(plain))
    return arabic >= 2 and arabic * 2 >= latin or 0 < arabic >= latin


def note(task):
    """What to add to a prompt so the model answers an Arabic task in Arabic; nothing for other tasks."""
    return "\n\n" + ANSWER_IN_ARABIC if written_in(task) else ""
