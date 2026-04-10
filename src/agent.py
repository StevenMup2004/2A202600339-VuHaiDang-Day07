from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self._store = store
        self._llm_fn = llm_fn
        self._max_chunk_chars = 4000
        self._max_context_chars = 12000

    def answer(self, question: str, top_k: int = 3) -> str:
        # Retrieve relevant chunks from the vector store
        retrieved = self._store.search(question, top_k=top_k)
        
        # Build context from retrieved chunks
        context_parts = []
        used_chars = 0
        for item in retrieved:
            chunk = item.get("content", "")
            if not chunk:
                continue

            chunk = chunk[: self._max_chunk_chars]
            remaining_budget = self._max_context_chars - used_chars
            if remaining_budget <= 0:
                break

            if len(chunk) > remaining_budget:
                chunk = chunk[:remaining_budget]

            context_parts.append(chunk)
            used_chars += len(chunk)
        
        context = "\n\n".join(context_parts)
        
        # Build the prompt with context
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Call the LLM to generate the answer
        answer = self._llm_fn(prompt)
        return answer
