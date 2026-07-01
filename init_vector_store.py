from qdrant_vector_store import QdrantVectorStoreWrapper


def get_vector_store() -> QdrantVectorStoreWrapper:
    """
    Factory function to return a singleton QdrantVectorStoreWrapper instance.
    """
    if not hasattr(get_vector_store, "_instance"):
        get_vector_store._instance = QdrantVectorStoreWrapper()
    return get_vector_store._instance
