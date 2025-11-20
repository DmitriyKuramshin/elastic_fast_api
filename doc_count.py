from elasticsearch import Elasticsearch

es = Elasticsearch(
    "http://10.3.3.16:9200"
)

for idx in ["flattened_hscodes_v4", "flattened_hscodes_v4_copy"]:
    resp = es.count(index=idx)
    print(f"{idx} -> {resp['count']}")
