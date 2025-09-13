# ingest.py
import pdfplumber, docx, pandas as pd
from io import BytesIO
from typing import List

def load_file(content: bytes, filename: str) -> str:
    """Extract text from PDF, DOCX, CSV, or TXT."""
    if filename.endswith(".pdf"):
        with pdfplumber.open(BytesIO(content)) as pdf:
            return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
    elif filename.endswith(".docx"):
        d = docx.Document(BytesIO(content))
        return "\n".join([p.text for p in d.paragraphs])
    elif filename.endswith(".csv"):
        df = pd.read_csv(BytesIO(content))
        return df.to_string()
    else:
        return content.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into smaller chunks for embeddings.
    
    :param text: Full text string
    :param chunk_size: Number of words per chunk
    :param overlap: Number of overlapping words between chunks
    :return: List of text chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move start by chunk_size - overlap
    return chunks


def process_file(content: bytes, filename: str) -> List[str]:
    """
    Load and chunk a document.
    
    :param content: File content in bytes
    :param filename: File name with extension
    :return: List of text chunks
    """
    full_text = load_file(content, filename)
    chunks = chunk_text(full_text)
    return chunks
