import os
from pathlib import Path
from .tools import add_to_knowledge_base

def ingest_directory(directory: Path, overwrite: bool = False):
    """
    Recursively ingest all text files in the given directory.
    """
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.lower().endswith((".txt", ".md", ".pdf")):
                file_path = Path(root) / file_name
                try:
                    if file_name.lower().endswith(".pdf"):
                        # Simple PDF text extraction
                        from PyPDF2 import PdfReader
                        reader = PdfReader(file_path)
                        content = ""
                        for page in reader.pages:
                            content += page.extract_text() or ""
                    else:
                        content = file_path.read_text(encoding="utf-8")
                    title = file_path.stem
                    add_to_knowledge_base(content=content, title=title)
                except Exception as e:
                    print(f"Failed to ingest {file_path}: {e}")
