import requests
from app.config import settings

HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{settings.EMBEDDING_MODEL_NAME}/pipeline/feature-extraction"

HEADERS = {
    "Authorization": f"Bearer {settings.HF_TOKEN}",
}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Get embeddings via HuggingFace's hosted Inference API. Processes in
    small batches with a generous timeout, since large models can take a
    while to cold-start on HF's side."""
    all_embeddings = []
    batch_size = 5  # smaller batches reduce risk of a single slow/failed call

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = requests.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": batch, "options": {"wait_for_model": True}},
            timeout=120,  # bge-m3 can be slow to cold-start on HF's side
        )

        if response.status_code != 200:
            raise RuntimeError(f"HF Inference API error {response.status_code}: {response.text}")

        embeddings = response.json()

        if isinstance(embeddings[0][0], list):
            import statistics
            for vec_group in embeddings:
                dim = len(vec_group[0])
                pooled_vec = [statistics.mean(token[d] for token in vec_group) for d in range(dim)]
                all_embeddings.append(pooled_vec)
        else:
            all_embeddings.extend(embeddings)

    return all_embeddings
