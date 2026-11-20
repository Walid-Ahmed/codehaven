"""Read the code in a picture, such as a screenshot, into text. Offline, with Tesseract on this computer.

Tesseract is a separate program the user installs once. CodeHaven runs it as a local command on a private
temporary copy of the picture: nothing is sent anywhere, and no picture or text is kept by this module.

Tesseract returns words with their positions. Indentation, which it otherwise drops, is rebuilt from those
positions, because code is drawn in a fixed-width font. Reading is never perfect, so the result names the
lines to check: words Tesseract was unsure of, characters that were repaired, and, for Python, the first
line that does not parse.
"""
import ast
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

from .workspace import MAX_BYTES

IMAGE_BYTES = 8_000_000
IMAGE_PIXELS = 40_000_000  # A small file can unpack to a huge picture; larger ones are refused unread.
SECONDS = 60
UNSURE = 60  # Tesseract's confidence, 0..100, below which a word's line is listed for checking.
CHECK_LINES = 40
# A picture named cart.js.png holds JavaScript; any other picture is saved as Python.
CODE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h", ".cpp", ".cs", ".go", ".rs", ".rb", ".php",
                 ".swift", ".kt", ".sh", ".sql", ".html", ".css", ".json", ".yaml", ".yml", ".toml", ".md", ".txt"}
INSTALL = ("Reading code from a picture needs Tesseract, a free program that runs on this computer. Install it once "
           "(macOS: brew install tesseract; Ubuntu or Debian: sudo apt install tesseract-ocr; Windows: the installer "
           "from the Tesseract project), then try again. After that it works offline.")


def image_suffix(data):
    """".png" or ".jpg" when the bytes are a picture CodeHaven reads, else None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    return None


def pixels(data):
    """Width times height from the picture's header, or None when the header cannot be read."""
    if data.startswith(b"\x89PNG") and len(data) >= 24:
        return int.from_bytes(data[16:20], "big") * int.from_bytes(data[20:24], "big")
    position = 2
    while position + 9 <= len(data) and data[position] == 0xFF:  # JPEG: walk the segments to the frame header.
        marker = data[position + 1]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return int.from_bytes(data[position + 5:position + 7], "big") * int.from_bytes(data[position + 7:position + 9], "big")
        position += 2 + int.from_bytes(data[position + 2:position + 4], "big")
    return None


def output_name(name):
    """The file name the text is saved under: shot.png becomes shot.py, and cart.js.png becomes cart.js."""
    stem = Path(name).stem
    return stem if Path(stem).suffix.lower() in CODE_SUFFIXES else stem + ".py"


def recognise(data, suffix):
    """Tesseract's table of words and positions for one picture."""
    program = shutil.which("tesseract")
    if program is None:
        raise ValueError(INSTALL)
    with tempfile.TemporaryDirectory(prefix="codehaven-ocr-") as folder:  # Readable by this user only.
        picture = Path(folder, "picture" + suffix)
        picture.write_bytes(data)
        # --psm 6 reads the picture as one block of text lines, which is what a screenshot of code is.
        command = [program, str(picture), "stdout", "--psm", "6", "-l", "eng", "tsv"]
        try:
            done = subprocess.run(command, capture_output=True, timeout=SECONDS, stdin=subprocess.DEVNULL, cwd=folder)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"Reading the picture took longer than {SECONDS} seconds and was stopped.") from exc
    if done.returncode != 0:
        last = [line.strip() for line in done.stderr.decode("utf-8", "replace").splitlines() if line.strip()]
        raise ValueError("Tesseract could not read the picture" + (f": {last[-1]}" if last else "."))
    return done.stdout.decode("utf-8", "replace")


def words(table):
    """Lines of words from Tesseract's table: each word is (left, top, width, height, confidence, text)."""
    lines, last = [], None
    for row in table.splitlines()[1:]:
        cells = row.split("\t", 11)  # Not the csv module: the text column holds bare quotes.
        if len(cells) < 12 or cells[0] != "5" or not cells[11].strip():
            continue
        key = tuple(cells[1:5])  # Page, block, paragraph, line.
        if key != last:
            lines.append([])
            last = key
        lines[-1].append((int(cells[6]), int(cells[7]), int(cells[8]), int(cells[9]), float(cells[10]), cells[11].strip()))
    return lines


def without_line_numbers(lines):
    """Drop an editor's line-number column: a first word on every line that counts upwards."""
    firsts = [line[0][5] for line in lines]
    if len(lines) < 3 or not all(first.isdigit() for first in firsts):
        return lines
    numbers = [int(first) for first in firsts]
    if any(later <= earlier for earlier, later in zip(numbers, numbers[1:])):
        return lines
    return [line[1:] for line in lines if len(line) > 1]


def character_width(lines):
    """Pixels per character of the fixed-width font, measured between the starts of neighbouring words."""
    rough = statistics.median(word[2] / len(word[5]) for line in lines for word in line)
    steps = [(after[0] - before[0]) / (len(before[5]) + 1) for line in lines for before, after in zip(line, line[1:])
             if after[0] - before[0] - before[2] < 2 * rough]  # Only pairs one space apart.
    return statistics.median(steps) if len(steps) >= 3 else rough


def layout(lines):
    """(text, unsure line numbers): the words put back where they stood, with indentation and blank lines."""
    lines = without_line_numbers(lines)
    if not lines:
        return "", []
    width = max(character_width(lines), 1.0)
    margin = min(line[0][0] for line in lines)
    # Blank lines are found from the distance between line centres; the commonest distance is one line.
    centres = [statistics.median(word[1] + word[3] / 2 for word in line) for line in lines]
    gaps = [later - earlier for earlier, later in zip(centres, centres[1:]) if later > earlier]
    pitch = statistics.median(gap for gap in gaps if gap < 1.5 * min(gaps)) if gaps else 0
    text, unsure = [], []
    for index, line in enumerate(lines):
        if index and pitch:
            text.extend([""] * min(3, max(0, round((centres[index] - centres[index - 1]) / pitch) - 1)))
        written = " " * max(0, round((line[0][0] - margin) / width)) + line[0][5]
        for before, after in zip(line, line[1:]):
            # Measured between the starts of the two words, so a narrow comma does not look like a wide gap.
            written += " " * max(1, round((after[0] - before[0]) / width) - len(before[5])) + after[5]
        text.append(written.rstrip())
        if any(word[4] < UNSURE for word in line):
            unsure.append(len(text))
    return "\n".join(text) + "\n", unsure


def parses(text):
    """None when the text is valid Python, else where it first goes wrong."""
    try:
        ast.parse(text)
    except SyntaxError as exc:
        return f"line {exc.lineno}: {exc.msg}"
    except (ValueError, RecursionError, MemoryError) as exc:
        return str(exc) or type(exc).__name__
    return None


def tidy(text, python):
    """(text, repaired line numbers): undo the misreadings that are safe to undo.

    Code has no curly quotes or long dashes, so those become plain ones. Coding fonts draw zero with a slash
    or a dot, which Tesseract reads as @; an @ that does not start a name, as in @property, is turned back
    into 0 unless that would break Python that parsed.
    """
    text = text.translate(str.maketrans("“”„‘’", "\"\"\"''"))
    text = re.sub(r"-?[—–]-?", "-", text)
    text = re.sub(r"(?m)(?<=[)\]\w]) :$", ":", text)  # A colon that ends a line was read as a word of its own.
    zero = re.sub(r"@(?![A-Za-z_])", "0", text)
    if zero == text or (python and parses(text) is None and parses(zero) is not None):
        return text, []
    repaired = [number for number, (old, new) in enumerate(zip(text.splitlines(), zero.splitlines()), 1) if old != new]
    return zero, repaired


def read_code(name, data, target=None):
    """Turn a picture of code into a text file, named after the picture unless target names it.

    Returns {"name", "text", "image", "lines", "check", "syntax"}: the new file's name and text, the picture's
    name, the number of lines read, the line numbers to compare with the picture, and for Python the first
    syntax error (None when it parses, or for other languages).
    """
    suffix = image_suffix(data)
    if suffix is None:
        raise ValueError("Only PNG and JPEG pictures can be read.")
    if len(data) > IMAGE_BYTES:
        raise ValueError(f"Only pictures up to {IMAGE_BYTES // 1_000_000} MB can be read.")
    size = pixels(data)
    if size is None or not 0 < size <= IMAGE_PIXELS:
        raise ValueError(f"Only pictures up to {IMAGE_PIXELS // 1_000_000} megapixels can be read.")
    text, unsure = layout(words(recognise(data, suffix)))
    if not text.strip():
        raise ValueError("No text was found in this picture. Use a sharp screenshot that shows only the code.")
    target = target or output_name(name)
    python = target.endswith(".py")
    text, repaired = tidy(text, python)
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError(f"The picture holds more than {MAX_BYTES // 1000} KB of text, more than CodeHaven reads.")
    return {"name": target, "text": text, "image": name, "lines": len(text.splitlines()),
            "check": sorted(set(unsure) | set(repaired))[:CHECK_LINES], "syntax": parses(text) if python else None}
