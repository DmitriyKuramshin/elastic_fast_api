import asyncio
from elasticsearch import AsyncElasticsearch

ES_HOST = "http://10.3.3.16:9200"
INDEX_NAME = "flattened_hscodes_v4_copy"
EMBEDDING_FIELDS = ["embedding_d1", "embedding_d2", "embedding_d3", "embedding_d4"]  # your 4 depth embeddings

async def count_embedded_docs():
    es = AsyncElasticsearch([ES_HOST])
    
    for field in EMBEDDING_FIELDS:
        query = {
            "query": {
                "exists": {
                    "field": field
                }
            }
        }
        resp = await es.count(index=INDEX_NAME, body=query)
        print(f"Documents with '{field}' embedding: {resp['count']}")
    
    await es.close()

if __name__ == "__main__":
    asyncio.run(count_embedded_docs())
