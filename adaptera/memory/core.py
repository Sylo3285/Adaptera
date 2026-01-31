"""
Adaptera Memory Core
-------------------
FAISS-backed vector database for storing embeddings with optional metadata.
Now supports file-backed persistence out-of-the-box and automatic embeddings via a small embedding model (SLM) using transformers.
"""

from __future__ import annotations
from typing import Any, List, Optional
import numpy as np
import faiss
import pickle
import os
import torch
from transformers import AutoTokenizer, AutoModel


class VectorDB:
    """
    FAISS-based vector database with persistent storage.

    - Stores embeddings and optional metadata.
    - Saves to disk automatically on changes.
    - Can automatically embed text using a small transformer model (SLM).
    """

    def __init__(
        self,
        dim: int = 384,  # matches MiniLM embedding size
        index_file: str = "memory.index",
        meta_file: str = "memory_meta.pkl",
        index_factory: str = "Flat",
        use_slm: bool = True,
        slm_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        """
        Args:
            dim: Dimensionality of embeddings.
            index_file: Path to save/load FAISS index.
            meta_file: Path to save/load metadata.
            index_factory: FAISS index type (e.g., "Flat", "IVF100,Flat").
            use_slm: Whether to use the small transformer model for automatic text embeddings.
            slm_model_name: Hugging Face model name for embedding.
            device: Device to run embeddings on (cuda or cpu)
        """
        self.dim = dim
        self.index_file = index_file
        self.meta_file = meta_file
        self.index_factory = index_factory

        # Load or create FAISS index
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
        else:
            self.index = faiss.index_factory(dim, index_factory)
            if not self.index.is_trained:
                self.index.train(np.zeros((1, dim), dtype=np.float32))

        # Load metadata
        if os.path.exists(meta_file):
            with open(meta_file, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata: List[Any] = []

        # Initialize small transformer embedding model if requested
        self.use_slm = use_slm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.use_slm:
            self.tokenizer = AutoTokenizer.from_pretrained(slm_model_name)
            self.model = AutoModel.from_pretrained(slm_model_name).to(self.device)
            self.model.eval()

    def save(self):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.metadata, f)

    @torch.inference_mode()
    def _embed_text(self, text: str) -> np.ndarray:
        """Convert text to embedding using the small transformer model."""
        if not self.use_slm:
            raise RuntimeError("SLM embedding is disabled, pass vectors manually.")

        # Tokenize and move to device
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_size)

        # Mean pooling
        mask = inputs['attention_mask'].unsqueeze(-1)  # (1, seq_len, 1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return pooled.cpu().numpy().astype(np.float32)  # shape (1, dim)

    def add(self, vectors: np.ndarray | str, metadata: Optional[List[Any]] = None):
        """
        Add vectors or raw text to the database with optional metadata.

        Args:
            vectors: either a numpy array (n, dim) or a single string (will be embedded)
            metadata: list of metadata objects, one per vector
        """
        # Auto-embed text if a string is passed
        if isinstance(vectors, str):
            text = vectors
            vectors = self._embed_text(vectors)
            metadata = metadata or [text]

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors of shape (n, {self.dim})")

        self.index.add(vectors)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([None] * len(vectors))

        self.save()  # persist to disk

    def search(self, query: np.ndarray | str, top_k: int = 5) -> list[tuple[float, Any]]:
        """Search for nearest vectors safely."""
        if isinstance(query, str):
            query = self._embed_text(query)  # only if embedding is implemented here

        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        distances, indices = self.index.search(query, top_k)

        results = []
        for dists, idxs in zip(distances, indices):
            result = []
            for d, i in zip(dists, idxs):
                if i == -1 or i >= len(self.metadata):
                    meta = None
                else:
                    meta = self.metadata[i]
                result.append((float(d), meta))
            results.append(result)

        return results if len(results) > 1 else results[0]



    def __len__(self):
        return self.index.ntotal
