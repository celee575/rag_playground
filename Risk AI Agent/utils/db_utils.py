from pathlib import Path
from typing import Iterable, Optional

import chromadb
from chromadb.config import Settings


# ──────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────

CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma"
LTM_COLLECTION_NAME = "ltm_embeddings"


def _get_chroma_collection(
    chroma_path: Optional[str | Path] = None,
) -> chromadb.Collection:
    """Return (or create) the ChromaDB collection for LTM embeddings."""
    resolved_chroma_path = Path(chroma_path) if chroma_path is not None else CHROMA_PATH
    resolved_chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(resolved_chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=LTM_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def ensure_ltm_vector_collection(
    chroma_path: Optional[str | Path] = None,
) -> chromadb.Collection:
    """Ensure the ChromaDB LTM collection exists and return it."""
    resolved_chroma_path = Path(chroma_path) if chroma_path is not None else CHROMA_PATH
    col = _get_chroma_collection(resolved_chroma_path)
    print(
        f"[LTM] ChromaDB collection '{LTM_COLLECTION_NAME}' ready at "
        f"{resolved_chroma_path}"
    )
    return col