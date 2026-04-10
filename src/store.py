from __future__ import annotations

import os
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        use_chroma = os.getenv("USE_CHROMA", "0").strip().lower() in {"1", "true", "yes"}
        if use_chroma:
            try:
                import chromadb

                client = chromadb.Client()
                self._collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                self._use_chroma = True
            except Exception:
                self._use_chroma = False
                self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # Embed the document content
        embedding = self._embedding_fn(doc.content)
        
        # Create and return record
        record = {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": embedding,
        }
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # Embed the query
        query_embedding = self._embedding_fn(query)
        
        # Compute similarities for all records
        similarities = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            similarities.append((record, score))
        
        # Sort by score descending and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        result = []
        for record, score in similarities[:top_k]:
            result_record = dict(record)
            result_record["score"] = score
            result.append(result_record)
        
        return result

    def _format_chroma_query_results(self, results: dict[str, Any]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        if not results or not results.get("documents"):
            return output

        ids = results.get("ids", [[]])
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
        distances = results.get("distances", [[]])

        for i, doc_content in enumerate(documents[0]):
            output.append(
                {
                    "id": ids[0][i] if ids and ids[0] else None,
                    "content": doc_content,
                    "metadata": metadatas[0][i] if metadatas and metadatas[0] else {},
                    # Keep descending-score semantics used by in-memory mode.
                    "score": 1.0 - float(distances[0][i]) if distances and distances[0] else 0.0,
                }
            )

        return output

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            record = self._make_record(doc)
            
            if self._use_chroma and self._collection is not None:
                # Add to ChromaDB
                try:
                    add_kwargs: dict[str, Any] = {
                        "ids": [doc.id],
                        "documents": [doc.content],
                        "embeddings": [record["embedding"]],
                    }
                    if doc.metadata:
                        add_kwargs["metadatas"] = [doc.metadata]
                    self._collection.add(**add_kwargs)
                except Exception:
                    # Graceful fallback if Chroma insert fails in local environments.
                    self._use_chroma = False
                    self._collection = None
                    self._store.append(record)
            else:
                # Add to in-memory store
                self._store.append(record)
            
            self._next_index += 1

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
            return self._format_chroma_query_results(results)
        else:
            # Use in-memory search
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                return len(self._store)
        else:
            return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            # No filter, just do regular search
            return self.search(query, top_k)
        
        if self._use_chroma and self._collection is not None:
            # Use ChromaDB with filter
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter
            )
            return self._format_chroma_query_results(results)
        else:
            # In-memory: filter first, then search
            filtered_records = []
            for record in self._store:
                # Check if all filter conditions match
                match = True
                for key, value in metadata_filter.items():
                    if record.get("metadata", {}).get(key) != value:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
            
            # Search among filtered records
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            try:
                # Try to get the document first to check if it exists
                doc = self._collection.get(ids=[doc_id])
                if doc and doc.get("ids"):
                    self._collection.delete(ids=[doc_id])
                    return True
                return False
            except Exception:
                return False
        else:
            # In-memory: remove all chunks with matching doc_id
            original_size = len(self._store)
            self._store = [r for r in self._store if r.get("id") != doc_id]
            return len(self._store) < original_size
