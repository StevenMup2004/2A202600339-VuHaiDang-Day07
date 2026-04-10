from __future__ import annotations

import hashlib
import math

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()
        # Initial safety cap; if still too long, __call__ will auto-shrink and retry.
        self._max_chars = 20000

    def __call__(self, text: str) -> list[float]:
        safe_text = text if len(text) <= self._max_chars else text[: self._max_chars]

        for _ in range(8):
            try:
                response = self.client.embeddings.create(model=self.model_name, input=safe_text)
                return [float(value) for value in response.data[0].embedding]
            except Exception as exc:
                message = str(exc).lower()
                is_context_error = (
                    "maximum context length" in message
                    or "invalid 'input'" in message
                    or "too many tokens" in message
                )
                if not is_context_error or len(safe_text) <= 1000:
                    raise

                # Reduce aggressively to guarantee convergence without tokenizer libs.
                safe_text = safe_text[: max(1000, len(safe_text) // 2)]

        # If the loop exits unexpectedly, surface a clear error.
        raise RuntimeError("Failed to generate embedding after adaptive truncation retries.")


_mock_embed = MockEmbedder()
