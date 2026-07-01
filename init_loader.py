import os
from pathlib import Path

from tools import add_to_knowledge_base
from init_vector_store import get_vector_store


def load_directory(path: str, overwrite: bool = False):
    """
    Walk through the specified directory, read all .txt files, and load them into the vector store.
    """
    if not os.path.isdir(path):
        raise ValueError(f"Provided path '{path}' is not a directory.")

    vector_store = get_vector_store()

    if overwrite:
        print("Deleting existing collection...")
        vector_store.delete_collection()
        print("Collection deleted. Recreating...")

    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".txt"):
                file_path = Path(root) / file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                title = file_path.stem
                source = str(file_path.resolve())
                result = add_to_knowledge_base(content, title, source)
                print(
                    f"Loaded '{title}' from '{source}'. Chunks added: {result['chunks_added']}"
                )

    # Persist after loading
    vector_store.persist()
    print("Vector store persisted to disk.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Bulk load text files into vector store.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the directory containing text files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing collection before loading.",
    )
    args = parser.parse_args()

    load_directory(args.path, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
