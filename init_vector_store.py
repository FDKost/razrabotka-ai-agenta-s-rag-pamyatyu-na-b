"""
Script to load all documents from a specified directory into the vector store.
"""

import argparse
from pathlib import Path

from init_loader import load_directory

def main():
    parser = argparse.ArgumentParser(description="Initialize the vector store with documents.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing .txt files to load.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing collection before loading.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.is_dir():
        raise ValueError(f"Data directory '{data_path}' does not exist.")

    load_directory(str(data_path), overwrite=args.overwrite)
    print("Vector store initialization complete.")

if __name__ == "__main__":
    main()
