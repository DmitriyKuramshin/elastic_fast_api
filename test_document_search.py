import json
from elasticsearch import Elasticsearch

es = Elasticsearch("http://10.3.3.16:9200")

# # Get settings
# settings = es.indices.get_settings(index="documents_v2")
# print("Settings for documents_v2:")
# print(json.dumps(settings["documents_v2"], indent=4))  # pretty-print
# print("\n" + "="*80 + "\n")

# # Get mappings
# mappings = es.indices.get_mapping(index="documents_v2")
# print("Mappings for documents_v2:")
# print(json.dumps(mappings["documents_v2"]["mappings"], indent=4))  # pretty-print

query = {
    "query": {
        "bool": {
            "should": [
                {
                    "match": {
                        "name": {
                            "query": "Hava yolu daşıma müqaviləsi",
                            "boost": 1
                        }
                    }
                },
                {
                    "match_phrase": {
                        "name": {
                            "query": "Hava yolu daşıma müqaviləsi",
                            "boost": 2
                        }
                    }
                },
                {
                    "term": {
                        "name.keyword": {
                            "value": "Hava yolu daşıma müqaviləsi",
                            "boost": 3
                        }
                    }
                }
            ]
        }
    }
}

res = es.search(index="documents_v2", body=query, size=10)  # size=10 for example
for hit in res['hits']['hits']:
    print(hit['_id'], hit['_source'])
