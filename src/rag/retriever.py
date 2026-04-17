"""
src/rag/retriever.py
====================
RAG (Retrieval-Augmented Generation) stub — ready for extension.

To activate:
  1. Install: pip install faiss-cpu sentence-transformers
  2. Build an index from EV knowledge documents
  3. Replace `retrieve()` body with real FAISS / ChromaDB lookup
  4. Inject retrieved context into agent.py's user prompt

Current behaviour: returns empty list (no-op).
"""

from __future__ import annotations

from typing import List


class EVKnowledgeRetriever:
    """
    Retrieves relevant EV infrastructure knowledge for a given query.

    Extension points:
        - Replace `_documents` with a real vector store
        - Implement `_embed()` using sentence-transformers
        - Use FAISS / ChromaDB for nearest-neighbour search
    """

    def __init__(self, index_path: str = "data/ev_knowledge_index"):
        """
        Args:
            index_path: Path to the vector index (not yet built).
        """
        self.index_path = index_path
        self._ready     = False  # Set to True once index is built

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the top-k most relevant knowledge snippets for query.

        Args:
            query:  Natural-language query (e.g. station ID + demand level)
            top_k:  Number of results to return

        Returns:
            List of relevant text snippets (empty list if index not ready).
        """
        if not self._ready:
            return []

        # TODO: implement vector search
        # embeddings = self._embed(query)
        # results    = self._index.search(embeddings, top_k)
        # return [self._documents[i] for i in results.indices[0]]
        return []

    # ------------------------------------------------------------------
    # Private helpers (stubs)
    # ------------------------------------------------------------------
    def _embed(self, text: str):
        """Embed text using sentence-transformers (not yet implemented)."""
        raise NotImplementedError("Install sentence-transformers and implement this.")

    def build_index(self, documents: List[str]) -> None:
        """
        Build the vector index from a list of text documents.

        Args:
            documents: List of EV knowledge text snippets
        """
        # TODO: embed documents and build FAISS index
        raise NotImplementedError("RAG index building not yet implemented.")
