# ingestion/local.py
import pathlib
from ingestion.chunker import chunk_text

def ingest_file(path: str) -> list[dict]:
    """Ingests a local file. Supports .txt, .md, .pdf."""
    p = pathlib.Path(path)

    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix in (".txt", ".md"):
        text = p.read_text(encoding="utf-8")

    elif p.suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        text = " ".join(
            page.extract_text() or "" for page in reader.pages
        )

    elif p.suffix == ".docx":
        from docx import Document
        doc = Document(str(p))
        text = " ".join(p.text for p in doc.paragraphs)

    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    return chunk_text(text, source=str(p))