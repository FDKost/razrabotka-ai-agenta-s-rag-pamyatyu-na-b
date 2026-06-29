import sys
from pathlib import Path
from src.ingestion import ingest_directory

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python init_db.py <directory>")
        sys.exit(1)
    directory = Path(sys.argv[1])
    ingest_directory(directory)
    print("Database initialized.")
