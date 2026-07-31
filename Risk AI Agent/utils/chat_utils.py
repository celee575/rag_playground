import json


# def compact_embedding(text: str) -> list[float]:
#     response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
#     embedding = response.embeddings[0].values
#     return [float(value) for value in embedding]

def parse_metadata_json_list(metadata, key):
    value = (metadata or {}).get(key)
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return [part.strip() for part in value.split(",") if part.strip()]
    return [value]


def chroma_hits(results):
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    return [
        {"id": item_id, "document": document, "metadata": metadata or {}, "distance": distance}
        for item_id, document, metadata, distance in zip(ids, documents, metadatas, distances)
    ]


def format_ltm_hit_for_chatbot(hit):
    metadata = hit["metadata"]
    return {
        "memory_type": "LTM",
        "id": hit["id"],
        "distance": hit["distance"],
        "session_id": metadata.get("session_id"),
        "summary": metadata.get("summary") or hit["document"],
        "topic_tags": parse_metadata_json_list(metadata, "topic_tags"),
    }





