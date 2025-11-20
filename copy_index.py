import asyncio
from elasticsearch import AsyncElasticsearch, helpers

ES_URL = "http://10.3.3.16:9200"
ORIGINAL_INDEX = "flattened_hscodes_v4"
COPY_INDEX = "flattened_hscodes_v4_copy"

settings_and_mappings = {
    "settings": {
        "max_ngram_diff": 21,
        "analysis": {
            "char_filter": {
                "az_html": {"type": "html_strip"}
            },
            "filter": {
                "az_stop": {
                    "type": "stop",
                    "stopwords": [
                        "və","ve","ilə","ile","ya","amma","lakin","çünki","cunki",
                        "halbuki","buna","bunu","bunlar","belə","bele","bəli","beli",
                        "xeyr","də","de","da","ki","bu","hər","her","heç","hec","nə",
                        "ne","niyə","niye","hara","burada","orada","hansı","hansi",
                        "necə","nece","çox","cox","az","bütün","butun","tək","tek",
                        "artıq","artiq","sonra","əgər","eger","olsun","olmaq","oldu",
                        "etmək","etmek","edir","edən","eden","haqqında","haqqinda",
                        "barədə","barede","isə","ise","deyil","daha","yenə","yene",
                        "indi","hələ","hele","yenidən","yeniden","üstdə","ustde",
                        "altında","altinda","arasında","arasinda","görə","gore",
                        "üzərinə","uzerine","qarşı","qarsi","baxmayaraq","məsələn",
                        "meselen","ancaq","çoxlu","coxlu","öz","oz","özünə","ozune",
                        "özünü","ozunu","özləri","ozleri","özüm","ozum","özümüz",
                        "ozumuz","özünüz","ozunuz","tərəfindən","terefinden","kimi",
                        "bununla","bəziləri","bezileri","heç kim","hec kim",
                        "hər kəs","her kes","heç nə","hec ne","yenə də","yene de"
                    ]
                }
            },
            "analyzer": {
                "az_index_plus": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "char_filter": ["az_html"],
                    "filter": ["lowercase", "az_stop"]
                },
                "az_search_plus": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "char_filter": ["az_html"],
                    "filter": ["lowercase", "az_stop"]
                },
                "az_search_plus_syn": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "char_filter": ["az_html"],
                    "filter": ["lowercase", "az_stop"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "code": {"type": "keyword"},
            "name_az_d1": {"type": "text", "analyzer": "az_index_plus", "search_analyzer": "az_search_plus_syn", "fields": {"raw": {"type": "keyword"}}},
            "name_az_d2": {"type": "text", "analyzer": "az_index_plus", "search_analyzer": "az_search_plus_syn", "fields": {"raw": {"type": "keyword"}}},
            "name_az_d3": {"type": "text", "analyzer": "az_index_plus", "search_analyzer": "az_search_plus_syn", "fields": {"raw": {"type": "keyword"}}},
            "name_az_d4": {"type": "text", "analyzer": "az_index_plus", "search_analyzer": "az_search_plus_syn", "fields": {"raw": {"type": "keyword"}}},
            "keywords_az_level1": {"type": "text", "analyzer": "az_index_plus", "search_analyzer": "az_search_plus_syn",
                                   "fields": {"raw": {"type": "keyword"}, "exact": {"type": "text", "analyzer": "az_index_plus", "search_analyzer": "az_search_plus_syn"}}},
            "keywords_az_level2": {"type": "text", "analyzer": "az_index_plus", "search_analyzer": "az_search_plus_syn",
                                   "fields": {"raw": {"type": "keyword"}, "exact": {"type": "text", "analyzer": "az_index_plus", "search_analyzer": "az_search_plus_syn"}}},
            "tradings": {"type": "nested", "properties": {"id": {"type": "keyword"},
                                                          "tradeType": {"type": "keyword"},
                                                          "tradeName": {"type": "text", "analyzer": "az_index_plus", "search_analyzer": "az_search_plus", "fields": {"raw": {"type": "keyword", "ignore_above": 256}}},
                                                          "inVehicleId": {"type": "long"},
                                                          "outVehicleId": {"type": "long"}}}
        }
    }
}

async def create_and_copy_index():
    es = AsyncElasticsearch(hosts=[ES_URL])
    try:
        # Delete if exists
        if await es.indices.exists(index=COPY_INDEX):
            await es.indices.delete(index=COPY_INDEX)
            print(f"Deleted existing index {COPY_INDEX}")
        
        # Create new index
        await es.indices.create(index=COPY_INDEX, body=settings_and_mappings)
        print(f"Created new index {COPY_INDEX} with settings/mappings.")
        
        # Scroll through original index and bulk insert into copy
        scroll_size = 500
        actions = []
        async for doc in helpers.async_scan(
            es, index=ORIGINAL_INDEX, query={"query": {"match_all": {}}}, size=scroll_size
        ):
            actions.append({
                "_op_type": "index",
                "_index": COPY_INDEX,
                "_id": doc["_id"],
                "_source": doc["_source"]
            })
            if len(actions) >= 500:
                await helpers.async_bulk(es, actions)
                actions = []
        
        if actions:
            await helpers.async_bulk(es, actions)
        
        print("All documents copied successfully.")
    
    finally:
        await es.close()

# Run the coroutine
asyncio.run(create_and_copy_index())
