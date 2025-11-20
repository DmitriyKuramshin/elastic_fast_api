import asyncio
from elasticsearch import AsyncElasticsearch, helpers
from sentence_transformers import SentenceTransformer

ES_URL = "http://10.3.3.16:9200"
INDEX = "flattened_hscodes_v4_copy"

# Load m12_1e model
MODEL_PATH = "m12_1e"
model = SentenceTransformer(MODEL_PATH)

async def generate_embeddings():
    es = AsyncElasticsearch(hosts=[ES_URL])
    try:
        scroll_size = 500
        actions = []

        async for doc in helpers.async_scan(
            es, index=INDEX, query={"query": {"match_all": {}}}, size=scroll_size
        ):
            source = doc["_source"]
            embeddings = {}

            # Generate embeddings for each depth
            for i in range(1, 5):
                field_name = f"name_az_d{i}"
                emb_field = f"embedding_d{i}"
                if source.get(field_name):
                    vec = model.encode(source[field_name])
                    embeddings[emb_field] = vec.tolist()

            if embeddings:
                action = {
                    "_op_type": "update",
                    "_index": INDEX,
                    "_id": doc["_id"],
                    "doc": embeddings
                }
                actions.append(action)

            # Bulk insert every 500 docs
            if len(actions) >= 500:
                await helpers.async_bulk(es, actions)
                actions = []

        # Final batch
        if actions:
            await helpers.async_bulk(es, actions)

        print("All embeddings generated and stored successfully.")

    finally:
        await es.close()

asyncio.run(generate_embeddings())
