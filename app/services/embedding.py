import requests
from app.config import settings

HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{settings.EMBEDDING_MODEL_NAME}"

HEADERS = {
    "Authorization": f"Bearer {settings.HF_TOKEN}",
}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Get embeddings via HuggingFace's hosted Inference API instead of loading
    the model locally — removes the memory load that was crashing Render's
    512MB instance."""
    response = requests.post(
        HF_API_URL,
        headers=HEADERS,
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(f"HF Inference API error {response.status_code}: {response.text}")

    embeddings = response.json()

    # The API can return either a single flat vector per text, or a nested
    # token-level structure depending on the model — normalize to one vector per input.
    if isinstance(embeddings[0][0], list):
        # Nested (token-level) output — mean-pool across tokens
        import statistics
        pooled = []
        for vec_group in embeddings:
            dim = len(vec_group[0])
            pooled_vec = [statistics.mean(token[d] for token in vec_group) for d in range(dim)]
            pooled.append(pooled_vec)
        return pooled

    return embeddings
