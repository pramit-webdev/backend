# ingest.py
import pdfplumber
import docx
import pandas as pd
from io import BytesIO
from typing import List


def load_file(content: bytes, filename: str) -> str:
    """Extract text from PDF, DOCX, CSV, or TXT."""
    filename = filename.lower()
    if filename.endswith(".pdf"):
        with pdfplumber.open(BytesIO(content)) as pdf:
            return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
    elif filename.endswith(".docx"):
        d = docx.Document(BytesIO(content))
        return "\n".join([p.text for p in d.paragraphs if p.text.strip()])
    elif filename.endswith(".csv"):
        df = pd.read_csv(BytesIO(content))
        return df.to_string(index=False)
    elif filename.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into smaller chunks for embeddings.
    Each chunk has 'chunk_size' words with 'overlap' overlap.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap  # move with overlap
    return chunks


def process_file(content: bytes, filename: str) -> List[str]:
    """
    Load and chunk a document.
    """
    full_text = load_file(content, filename)
    return chunk_text(full_text)
