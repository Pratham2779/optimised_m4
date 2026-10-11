"""Optimized LLM client: batch embeddings, async, caching, retry."""
import re
import asyncio
from typing import List, Dict, Any, Optional
import httpx
import orjson
from async_lru import alru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.config import AppConfig, setup_logger
from src.models import RawMemoryItem

logger = setup_logger("m4.llm_client")

_CODE_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
_BRACKET_PATTERN = re.compile(r"\[.*\]", re.DOTALL)

class LLMClientException(Exception):
    pass

class OllamaClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._embedding_cache: Dict[str, List[float]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=300.0,
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @alru_cache(maxsize=4096)
    async def get_embedding(self, text: str) -> tuple:
        """Single embedding with async LRU cache. Returns tuple (hashable)."""
        if text in self._embedding_cache:
            return tuple(self._embedding_cache[text])
        client = await self._get_client()
        base = str(self.config.ollama_base_url).rstrip('/').replace("localhost", "127.0.0.1")
        url = f"{base}/api/embed"
        payload = {"model": self.config.embedding_model, "input": text}
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = orjson.loads(resp.content)
        emb = tuple(data["embeddings"][0])
        self._embedding_cache[text] = list(emb)
        return emb

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Optimized batch embeddings with in-memory caching for repeated texts."""
        if not texts:
            return []

        # Identify unique cache misses
        missing = [t for t in dict.fromkeys(texts) if t not in self._embedding_cache]
        if missing:
            client = await self._get_client()
            base = str(self.config.ollama_base_url).rstrip('/').replace("localhost", "127.0.0.1")
            url = f"{base}/api/embed"
            payload = {"model": self.config.embedding_model, "input": missing}
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = orjson.loads(resp.content)
            for text, emb in zip(missing, data["embeddings"]):
                self._embedding_cache[text] = list(emb)

        return [self._embedding_cache[t] for t in texts]


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        reraise=True,
    )
    async def complete_chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        client = await self._get_client()
        base = str(self.config.ollama_base_url).rstrip('/').replace("localhost", "127.0.0.1")
        url = f"{base}/v1/chat/completions"
        payload = {"model": self.config.llm_model, "messages": messages, "temperature": temperature}
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = orjson.loads(resp.content)
        return data["choices"][0]["message"]["content"]

    async def extract_structured_json(self, prompt: str) -> List[RawMemoryItem]:
        messages = [
            {"role": "system", "content": "You are a memory extraction engine. Respond strictly with a JSON array."},
            {"role": "user", "content": prompt},
        ]
        raw_text = await self.complete_chat(messages, temperature=0.0)
        parsed = self._parse_json_array(raw_text)
        if parsed is not None:
            return parsed
        retry_messages = [
            {"role": "system", "content": "Output ONLY valid JSON array with keys 'Category' and 'Summary'."},
            {"role": "user", "content": f"Invalid JSON:\n{raw_text}\nFix it."},
        ]
        second = await self.complete_chat(retry_messages, temperature=0.0)
        parsed2 = self._parse_json_array(second)
        if parsed2 is not None:
            return parsed2
        raise LLMClientException("Failed to extract valid JSON after retry.")

    def _parse_json_array(self, text: str) -> Optional[List[RawMemoryItem]]:
        clean = text.strip()
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", clean, re.DOTALL)
        if m:
            clean = m.group(1)
        else:
            m2 = re.search(r"\[.*\]", clean, re.DOTALL)
            if m2:
                clean = m2.group(0)
        try:
            data = orjson.loads(clean)
            if not isinstance(data, list):
                return None
            return [RawMemoryItem.model_validate(item) for item in data]
        except Exception:
            return None
