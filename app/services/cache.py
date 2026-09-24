from cachetools import TTLCache
import hashlib

# Cache up to 200 recent query results, each expires after 1 hour
query_cache = TTLCache(maxsize=200, ttl=3600)

def make_cache_key(question: str, regime: str, top_k: int) -> str:
    raw = f"{question.strip().lower()}|{regime}|{top_k}"
    return hashlib.sha256(raw.encode()).hexdigest()