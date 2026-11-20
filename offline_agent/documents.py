"""Read a Jupyter notebook or a PDF into one text file, so it can be chatted about like any uploaded file.

Both are read on this computer and nothing leaves it. The original file is never changed: the chat is about a
text copy kept in the same private temporary folder as any uploaded file.

A notebook (.ipynb) is JSON. Its code and Markdown cells become one file in the "percent" layout editors
understand: each cell starts with a "# %%" line that carries its number, and Markdown becomes comments.
Outputs, pictures, and widget state are left out. A Python notebook becomes a .py file.

A PDF's text is read by pypdf when that package is installed, or else by the pdftotext program (part of
Poppler). Either works offline. Each page starts with a "===== Page N =====" line, so a cited line can be
found on its page. A scanned PDF holds pictures, not text; a screenshot of the page can be read instead.

A document longer than CodeHaven reads at once (64 KB of text) is cut at a cell or page boundary, and the
result says how much was kept.
"""
import io
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .workspace import MAX_BYTES

DOCUMENT_BYTES = 8_000_000  # The largest notebook or PDF accepted; the text read from it is far smaller.
PDF_PAGES, PDF_SECONDS = 300, 60
EXTENSIONS = {"python": ".py", "javascript": ".js", "typescript": ".ts", "r": ".r", "julia": ".jl", "ruby": ".rb",
              "scala": ".scala", "bash": ".sh", "go": ".go", "rust": ".rs", "java": ".java", "kotlin": ".kt"}
PDF_HELP = ("Reading a PDF needs pypdf or pdftotext on this computer, and neither was found. Install one once while "
            "online: `python -m pip install pypdf`, or Poppler (`brew install poppler`; Ubuntu or Debian: "
            "`sudo apt install poppler-utils`). Everything else keeps working.")


def kind(name, data):
    """"pdf", "notebook", or None for any other upload."""
    if data[:1024].lstrip().startswith(b"%PDF-"):
        return "pdf"
    if str(name).lower().endswith(".ipynb"):
        return "notebook"
    return None


def stem(name):
    base = re.sub(r"\.(?:ipynb|pdf)$", "", name, flags=re.I)
    return base if base.strip("._") else "document"


def source(cell):
    text = cell.get("source", "")
    text = "".join(part for part in text if isinstance(part, str)) if isinstance(text, list) else text
    return text if isinstance(text, str) else ""


def read_notebook(name, data):
    """Turn a notebook into one text file. Returns {"name", "text", "source", "kind", "note"}."""
    if len(data) > DOCUMENT_BYTES:
        raise ValueError(f"Only notebooks up to {DOCUMENT_BYTES // 1_000_000} MB can be read.")
    try:
        notebook = json.loads(data.decode("utf-8"))
        cells = notebook["cells"] if isinstance(notebook, dict) else None
    except (ValueError, KeyError, RecursionError):
        cells = None
    if not isinstance(cells, list):
        raise ValueError("This does not look like a Jupyter notebook (.ipynb): it holds no list of cells.")
    metadata = notebook.get("metadata") if isinstance(notebook.get("metadata"), dict) else {}
    language = ""
    for section in ("kernelspec", "language_info"):
        info = metadata.get(section)
        if isinstance(info, dict) and isinstance(info.get("language") or info.get("name"), str):
            language = (info.get("language") or info.get("name")).lower()
            break
    extension = next((ext for known, ext in EXTENSIONS.items() if language.startswith(known)), ".py" if not language else ".txt")
    comment = "//" if extension in (".js", ".ts", ".go", ".rs", ".java", ".kt", ".scala") else "#"
    blocks, kept, code_cells, size = [], 0, 0, 0
    usable = [cell for cell in cells if isinstance(cell, dict) and cell.get("cell_type") in ("code", "markdown", "raw")
              and source(cell).strip()]
    for number, cell in enumerate(usable, 1):
        body = source(cell).rstrip("\n")
        if cell["cell_type"] == "code":
            # IPython magics and shell lines are not valid code in the file, so they are kept as comments.
            body = "\n".join(f"{comment} {line}" if re.match(r"\s*[%!]", line) else line for line in body.splitlines())
            block = f"{comment} %% Cell {number}\n{body}\n"
        else:
            block = (f"{comment} %% [markdown] Cell {number}\n"
                     + "\n".join(f"{comment} {line}".rstrip() for line in body.splitlines()) + "\n")
        if size + len(block.encode()) + 1 > MAX_BYTES:
            break
        blocks.append(block)
        size += len(block.encode()) + 1
        kept += 1
        code_cells += cell["cell_type"] == "code"
    if not blocks:
        raise ValueError("This notebook has no code or text cells to read."
                         if not usable else f"The notebook's first cell alone is over {MAX_BYTES // 1000} KB, more than CodeHaven reads.")
    note = (f"Read {kept} cell{'s' if kept != 1 else ''} ({code_cells} with code) from the notebook {name}. "
            "Outputs and pictures are left out; each cell starts with a line naming its number.")
    if kept < len(usable):
        note += f" Only cells 1-{kept} of {len(usable)} fit in the {MAX_BYTES // 1000} KB CodeHaven reads at once."
    return {"name": stem(name) + extension, "text": "\n".join(blocks), "source": name, "kind": "notebook", "note": note}


def pdf_pages(data):
    """The text of each page, in order, read by pypdf when installed and otherwise by pdftotext."""
    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None
    if PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("This PDF is protected by a password, so its text cannot be read.")
            pages = []
            for page in reader.pages[:PDF_PAGES]:
                try:
                    pages.append(page.extract_text(extraction_mode="layout") or "")
                except Exception:  # One odd page should not lose the document.
                    pages.append(page.extract_text() or "")
            return pages, len(reader.pages)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"This PDF could not be read: {str(exc)[:160] or type(exc).__name__}") from exc
    program = shutil.which("pdftotext")
    if program is None:
        raise ValueError(PDF_HELP)
    with tempfile.TemporaryDirectory(prefix="codehaven-pdf-") as folder:  # Readable by this user only; deleted after.
        copy = Path(folder, "document.pdf")
        copy.write_bytes(data)
        try:
            done = subprocess.run([program, "-layout", "-enc", "UTF-8", "-l", str(PDF_PAGES), str(copy), "-"],
                                  capture_output=True, stdin=subprocess.DEVNULL, timeout=PDF_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Reading this PDF took too long and was stopped.") from exc
        except OSError as exc:
            raise ValueError(f"pdftotext could not be started: {exc}") from exc
    if done.returncode != 0:
        last = [line.strip() for line in done.stderr.decode("utf-8", "replace").splitlines() if line.strip()]
        raise ValueError("This PDF could not be read" + (f": {last[-1][:160]}" if last else "."))
    pages = done.stdout.decode("utf-8", "replace").split("\f")
    if pages and not pages[-1].strip():
        pages.pop()  # pdftotext ends every page, the last one too, with a form feed.
    return pages, len(pages)


def read_pdf(name, data):
    """Turn a PDF's text into one text file. Returns {"name", "text", "source", "kind", "note"}."""
    if len(data) > DOCUMENT_BYTES:
        raise ValueError(f"Only PDFs up to {DOCUMENT_BYTES // 1_000_000} MB can be read.")
    pages, total = pdf_pages(data)
    blocks, kept, size = [], 0, 0
    for number, page in enumerate(pages, 1):
        lines = [line.rstrip() for line in page.replace("\0", "").splitlines()]
        while lines and not lines[-1]:
            lines.pop()
        while lines and not lines[0]:
            lines.pop(0)
        # Layout mode pads every line alike; drop the shared margin but keep the indentation code depends on.
        margin = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
        block = f"===== Page {number} =====\n" + "\n".join(line[margin:] if line.strip() else "" for line in lines) + "\n"
        if size + len(block.encode()) + 1 > MAX_BYTES:
            break
        blocks.append(block)
        size += len(block.encode()) + 1
        kept += 1
    if not any(block.split("\n", 1)[1].strip() for block in blocks):
        raise ValueError("No text was found in this PDF. It may be scanned pages, which are pictures: attach a "
                         "screenshot of the page instead and CodeHaven reads the code in it."
                         if pages else "This PDF has no pages to read.")
    note = (f"Read {kept} page{'s' if kept != 1 else ''} of text from the PDF {name}. Each page starts with a line naming "
            "its number. Columns, tables, and code can lose their layout, so check anything that matters against the PDF.")
    if kept < total:
        note += f" Only pages 1-{kept} of {total} fit in the {MAX_BYTES // 1000} KB CodeHaven reads at once."
    return {"name": stem(name) + ".txt", "text": "\n".join(blocks), "source": name, "kind": "pdf", "note": note}


def read_document(name, data):
    found = kind(name, data)
    if found is None:
        raise ValueError("Only notebooks (.ipynb) and PDFs can be read as documents.")
    return read_pdf(name, data) if found == "pdf" else read_notebook(name, data)
