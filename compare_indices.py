import asyncio
import json
from elasticsearch import AsyncElasticsearch

ES_URL = "http://10.3.3.16:9200"
ORIGINAL_INDEX = "flattened_hscodes_v4"
COPY_INDEX = "flattened_hscodes_v4_copy"

async def compare_mappings():
    es = AsyncElasticsearch(hosts=[ES_URL])
    
    try:
        # Get mappings
        original_mapping = await es.indices.get_mapping(index=ORIGINAL_INDEX)
        copy_mapping = await es.indices.get_mapping(index=COPY_INDEX)
        
        # Extract actual mappings dicts
        orig_map = original_mapping[ORIGINAL_INDEX]["mappings"]
        copy_map = copy_mapping[COPY_INDEX]["mappings"]
        
        print("=== Original Index Mappings ===")
        print(json.dumps(orig_map, indent=2))
        
        print("\n=== Copy Index Mappings ===")
        print(json.dumps(copy_map, indent=2))
        
        # Compare keys
        orig_fields = set(orig_map.get("properties", {}).keys())
        copy_fields = set(copy_map.get("properties", {}).keys())
        
        only_in_original = orig_fields - copy_fields
        only_in_copy = copy_fields - orig_fields
        
        print("\nFields only in original:", only_in_original)
        print("Fields only in copy:", only_in_copy)
        
        # Check for differences in field types
        common_fields = orig_fields & copy_fields
        differences = {}
        for f in common_fields:
            orig_type = orig_map["properties"][f].get("type")
            copy_type = copy_map["properties"][f].get("type")
            if orig_type != copy_type:
                differences[f] = {"original": orig_type, "copy": copy_type}
        
        print("\nFields with different types:", differences)
    
    finally:
        await es.close()

asyncio.run(compare_mappings())
