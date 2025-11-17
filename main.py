from enum import Enum
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from elasticsearch import AsyncElasticsearch
from sentence_transformers import SentenceTransformer
import re
import asyncio

# ==================== Data Models ====================

class TradingType(str, Enum):
    IMPORT = "IMPORT"
    EXPORT = "EXPORT"
    TRANSIT = "TRANSIT"


class VehicleType(str, Enum):
    demiryolu = "demiryolu"  # ID: 3 - Dəmiryolu nəqliyyatı
    deniz = "deniz"          # ID: 1 - Dəniz nəqliyyatı
    avtomobil = "avtomobil"  # ID: 2 - Avtomobil nəqliyyatı
    hava = "hava"            # ID: 4 - Hava nəqliyyatı
    
    def to_id(self) -> int:
        """Convert vehicle type to numeric ID for Elasticsearch"""
        mapping = {
            "deniz": 1,
            "avtomobil": 2,
            "demiryolu": 3,
            "hava": 4
        }
        return mapping[self.value]


class Trading(BaseModel):
    id: Optional[str] = None
    tradeType: str
    tradeName: str
    inVehicleId: Optional[int] = None
    outVehicleId: Optional[int] = None


class Filter(BaseModel):
    model_config = ConfigDict(extra="ignore")
    trade_type: Optional[TradingType] = Field(default=None, description="Single trade type filter")
    in_vehicle_ids: List[VehicleType] = Field(default_factory=list, description="Incoming vehicle IDs (applies with trade_type)")
    out_vehicle_ids: List[VehicleType] = Field(default_factory=list, description="Outgoing vehicle IDs (applies with trade_type)")


class ElasticDocument(BaseModel):
    id: str
    code: Optional[str] = None
    score: Optional[float] = None
    name_az_d1: Optional[str] = None
    name_az_d2: Optional[str] = None
    name_az_d3: Optional[str] = None
    name_az_d4: Optional[str] = None
    tradings: List[Trading] = Field(default_factory=list)
    Path: Optional[str] = None

    @staticmethod
    def build_path(p1, p2, p3):
        parts = [p for p in (p1, p2, p3) if p]
        return " / ".join(parts)

    @staticmethod
    def from_es_hit(hit: dict) -> "ElasticDocument":
        src = hit.get("_source") or {}
        score = hit.get("_score")
        tradings_data = src.get("tradings", [])

        tradings = []
        if tradings_data:
            for t in tradings_data:
                if isinstance(t, dict):
                    try:
                        tradings.append(Trading(**t))
                    except Exception:
                        continue
        
        path = ElasticDocument.build_path(
            src.get("name_az_d1"), 
            src.get("name_az_d2"), 
            src.get("name_az_d3")
        )
        return ElasticDocument(
            id=src.get("id") or hit.get("_id"),
            code=src.get("code"),
            score=score,
            name_az_d1=src.get("name_az_d1"),
            name_az_d2=src.get("name_az_d2"),
            name_az_d3=src.get("name_az_d3"),
            name_az_d4=src.get("name_az_d4"),
            tradings=tradings,
            Path=path or src.get("Path"),
        )


class HybridRetrievedResponseSet(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    query_text: str = Field(alias="query-text")
    total_hits: int = Field(alias="total-hits")
    ranked_objects: List[ElasticDocument] = Field(
        default_factory=list, 
        alias="Ranked-objects"
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    filter: Filter = Field(default_factory=Filter)
    size: int = Field(default=10, ge=1, le=200, description="Number of results to return (same as top_k)")
    top_k: Optional[int] = Field(default=None, ge=1, le=200, description="Alias for size parameter (number of top results)")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for vector similarity (0=BM25 only, 1=vector only)")
    use_vector: bool = Field(default=True, description="Enable vector search in hybrid mode")
    
    def model_post_init(self, __context):
        """If top_k is provided, use it as size"""
        if self.top_k is not None:
            self.size = self.top_k


# ==================== Query Builder ====================

import re

def build_es_bool_query(req: SearchRequest) -> dict:
    f = req.filter

    query_text = req.query.strip()
    
    numeric_code = re.sub(r'\D', '', query_text)
    has_code = len(numeric_code) >= 6
    
    is_only_code = query_text.isdigit() and len(query_text) in [6, 8, 10]
    
    should_clauses = []
    
    parent_coefficient = 0.3
    
    # If query is only a code, use exact prefix matching only
    if is_only_code:
        should_clauses.append({
            "prefix": {
                "code": {
                    "value": query_text,
                    "boost": 10.0
                }
            }
        })
    else:
        # 1) Exact match on raw field for name_az_d4
        should_clauses.append({
            "term": {
                "name_az_d4.raw": {
                    "value": query_text,
                    "boost": 12.0
                }
            }
        })

        # 2) Main fuzzy match for name_az_d4 with prefix_length
        should_clauses.append({
            "match": {
                "name_az_d4": {
                    "query": req.query,
                    "boost": 2.0,
                    "fuzziness": "AUTO",
                    "prefix_length": 2
                }
            }
        })
        
        # 3) Prefix query on name_az_d4 (kept from original)
        should_clauses.append({
            "prefix": {
                "name_az_d4": {
                    "value": req.query,
                    "boost": 5.0
                }
            }
        })
        
        # 4) Phrase match for exact phrase matching
        should_clauses.append({
            "match_phrase": {
                "name_az_d4": {
                    "query": req.query,
                    "boost": 5.0
                }
            }
        })

        # Add code prefix queries with tiered boosting if numeric code is detected
        if has_code:
            code_len = len(numeric_code)
            
            # Level 1: 6+ characters (base boost)
            should_clauses.append({
                "prefix": {
                    "code": {
                        "value": numeric_code[:6],
                        "boost": 10.0
                    }
                }
            })
            
            # Level 2: 8+ characters (higher boost)
            if code_len >= 8:
                should_clauses.append({
                    "prefix": {
                        "code": {
                            "value": numeric_code[:8],
                            "boost": 20.0
                        }
                    }
                })
            
            # Level 3: 10+ characters (highest boost)
            if code_len >= 10:
                should_clauses.append({
                    "prefix": {
                        "code": {
                            "value": numeric_code[:10],
                            "boost": 30.0
                        }
                    }
                })

        # Parent fields with coefficient applied to their boosting
        parent_fields = [
            ("name_az_d1", 1.0),
            ("name_az_d2", 1.5),
            ("name_az_d3", 2.0),
        ]
        
        for field, boost in parent_fields:
            # 1) Exact match on raw field for parent names
            should_clauses.append({
                "term": {
                    f"{field}.raw": {
                        "value": query_text,
                        "boost": 6.0 * boost * parent_coefficient
                    }
                }
            })

            # 2) Fuzzy match with prefix_length
            should_clauses.append({
                "match": {
                    field: {
                        "query": req.query,
                        "boost": boost * parent_coefficient,
                        "fuzziness": "AUTO",
                        "prefix_length": 2
                    }
                }
            })

            # 3) Prefix query on parent fields (kept from original)
            should_clauses.append({
                "prefix": {
                    field: {
                        "value": req.query,
                        "boost": 2.5 * boost * parent_coefficient
                    }
                }
            })
    
    bool_query = {
        "should": should_clauses,
        "minimum_should_match": 1
    }
    
    filters = []
    
    # Apply filters with AND logic: trade_type AND vehicle_ids must match together
    # If trade_type is specified, we need nested query with must conditions
    if f.trade_type or f.in_vehicle_ids or f.out_vehicle_ids:
        # Build nested must conditions for the same trading object
        nested_must = []
        
        # Trade type must match
        if f.trade_type:
            nested_must.append({
                "term": {"tradings.tradeType": f.trade_type.value}
            })
        
        # Build vehicle filters
        # If both in and out vehicle filters exist, they both must match
        if f.in_vehicle_ids:
            nested_must.append({
                "terms": {"tradings.inVehicleId": [v.to_id() for v in f.in_vehicle_ids]}
            })
        
        if f.out_vehicle_ids:
            nested_must.append({
                "terms": {"tradings.outVehicleId": [v.to_id() for v in f.out_vehicle_ids]}
            })
        
        # Add nested query that ensures all conditions match within the same nested object
        if nested_must:
            filters.append({
                "nested": {
                    "path": "tradings",
                    "query": {
                        "bool": {
                            "must": nested_must
                        }
                    }
                }
            })
    
    if filters:
        bool_query["filter"] = filters
    
    query = {
        "function_score": {
            "query": {
                "bool": bool_query
            },
            "functions": [],
            "score_mode": "multiply",
            "boost_mode": "multiply"
        }
    }
    
    return query


def build_hybrid_vector_query(req: SearchRequest, query_vector: List[float]) -> dict:
    """Build hybrid query combining BM25 and vector similarity with filters"""
    f = req.filter
    
    # Build the base BM25 query (without function_score wrapper)
    base_query = build_es_bool_query(req)
    # Extract the bool query from function_score
    bool_query = base_query["function_score"]["query"]["bool"]
    
    # Build script_score query for hybrid search
    query = {
        "script_score": {
            "query": {
                "bool": bool_query
            },
            "script": {
                "source": """
                double bm25 = _score / (_score + 10.0);
                double cosine = (cosineSimilarity(params.query_vector, 'embedding') + 1.0) / 2.0;
                return params.alpha * cosine + (1 - params.alpha) * bm25;
                """,
                "params": {
                    "query_vector": query_vector,
                    "alpha": req.alpha
                }
            }
        }
    }
    
    return query


# ==================== FastAPI Application ====================

# Source Elasticsearch (for BM25 search)
# SOURCE_ES_URL = "http://10.3.3.16:9200"
SOURCE_ES_URL = "https://c70506d900e44618bd38984c5803a2ea.us-central1.gcp.cloud.es.io:443"
SOURCE_ES_API_KEY = "VDVCdlZab0J4Q0dCdlkwSTJEOEg6aFNud3BhSkN0QWxRR1dmVVpCdkotdw=="
SOURCE_ES_INDEX = "flattened_hscodes_v2"

# Destination Elasticsearch (for vector search)
DEST_ES_URL = "https://8b64d8075c244822b5b7d37c8326f96f.us-central1.gcp.cloud.es.io:443"
DEST_ES_API_KEY = "Ql9uY1ZKb0JCd0t3WlRBbUo1LUk6V25TQmprTlpUR042dFoxMS1Tb0NPdw=="
DEST_ES_INDEX = "embedded_words"

MODEL_DIR = "m12_1e"  # Path to your sentence transformer model

app = FastAPI(
    title="Hybrid Search API with Vector Search",
    version="2.0.0",
    description="FastAPI service for hybrid search with Elasticsearch and vector embeddings"
)


def load_model():
    """Load the sentence transformer model"""
    print(f"Loading model from: {MODEL_DIR}")
    try:
        model = SentenceTransformer(MODEL_DIR)
        print(f"✅ Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as e:
        print(f"⚠️ Warning: Could not load model from {MODEL_DIR}: {e}")
        print("Vector search will be disabled.")
        return None


@app.on_event("startup")
async def startup():
    """Initialize Elasticsearch connections and load model on startup"""
    # Connect to source ES for BM25 search
    app.state.es_source = AsyncElasticsearch(hosts=[SOURCE_ES_URL], api_key=SOURCE_ES_API_KEY)
    print(f"✅ Connected to Source Elasticsearch at {SOURCE_ES_URL}")
    
    # Connect to destination ES for vector search
    app.state.es_dest = AsyncElasticsearch(hosts=[DEST_ES_URL], api_key=DEST_ES_API_KEY)
    print(f"✅ Connected to Destination Elasticsearch at {DEST_ES_URL}")
    
    # Load the embedding model
    app.state.model = load_model()


@app.on_event("shutdown")
async def shutdown():
    """Close Elasticsearch connections on shutdown"""
    await app.state.es_source.close()
    await app.state.es_dest.close()
    print("Elasticsearch connections closed")


@app.post(
    "/search",
    response_model=HybridRetrievedResponseSet,
    response_model_by_alias=True,
    summary="Hybrid Search with Vector Embeddings",
    description="Perform hybrid search combining BM25 and vector similarity"
)
async def search(req: SearchRequest) -> HybridRetrievedResponseSet:
    """
    Execute a hybrid search query against Elasticsearch.
    
    Args:
        req: SearchRequest containing query text, filters, size/top_k, alpha, and use_vector flag
        
    Returns:
        HybridRetrievedResponseSet with ranked search results
    """
    
    # Check if vector search is enabled and model is available
    use_vector_search = req.use_vector and app.state.model is not None
    
    if use_vector_search:
        # Generate query embedding (run in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        query_vector = await loop.run_in_executor(
            None, 
            lambda: app.state.model.encode(req.query).tolist()
        )
        
        # Build hybrid query with vector similarity
        es_query = build_hybrid_vector_query(req, query_vector)
        
        # Use destination ES with embeddings
        try:
            resp = await app.state.es_dest.search(
                index=DEST_ES_INDEX,
                query=es_query,
                size=req.size
            )
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Elasticsearch (vector) error: {e}"
            )
        
        # Parse hits from vector search
        hits_data = (resp or {}).get("hits", {})
        hits = hits_data.get("hits", []) or []
        total_hits = hits_data.get("total", {})
        
        # Extract codes and scores from vector search results
        vector_results = {}
        for hit in hits:
            code = hit.get("_source", {}).get("code")
            score = hit.get("_score")
            if code:
                vector_results[code] = score
        
        # Fetch full documents from source ES using codes
        if vector_results:
            try:
                # Build a query to fetch documents by codes
                codes_list = list(vector_results.keys())
                fetch_resp = await app.state.es_source.search(
                    index=SOURCE_ES_INDEX,
                    query={
                        "terms": {
                            "code": codes_list
                        }
                    },
                    size=len(codes_list)
                )
                
                # Build a map of code -> full document
                full_docs = {}
                for hit in fetch_resp.get("hits", {}).get("hits", []):
                    code = hit.get("_source", {}).get("code")
                    if code:
                        # Replace the score with the vector search score
                        hit["_score"] = vector_results.get(code, 0.0)
                        full_docs[code] = hit
                
                # Rebuild hits in the order of vector search results
                enriched_hits = []
                for code in codes_list:
                    if code in full_docs:
                        enriched_hits.append(full_docs[code])
                
                hits = enriched_hits
                
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Elasticsearch (source fetch) error: {e}"
                )
    else:
        # Build standard BM25 query
        es_query = build_es_bool_query(req)
        
        # Use source ES for BM25 search
        try:
            resp = await app.state.es_source.search(
                index=SOURCE_ES_INDEX,
                query=es_query,
                size=req.size
            )
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Elasticsearch (BM25) error: {e}"
            )
        
        # Parse hits and total count
        hits_data = (resp or {}).get("hits", {})
        hits = hits_data.get("hits", []) or []
        total_hits = hits_data.get("total", {})
    
    # Extract total count (Elasticsearch can return this in different formats)
    if isinstance(total_hits, dict):
        total_count = total_hits.get("value", 0)
    else:
        total_count = total_hits or 0
    
    ranked = [ElasticDocument.from_es_hit(h) for h in hits]
    
    # Return response with all required fields
    return HybridRetrievedResponseSet(
        **{
            "query-text": req.query, 
            "total-hits": total_count,
            "Ranked-objects": ranked
        }
    )


@app.get("/health", summary="Health Check")
async def health_check():
    """Check if the service, Elasticsearch connections, and model are healthy"""
    try:
        es_source_health = await app.state.es_source.info()
        es_dest_health = await app.state.es_dest.info()
        model_status = "loaded" if app.state.model is not None else "not loaded"
        
        return {
            "status": "healthy",
            "elasticsearch_source": {
                "status": "connected",
                "cluster_name": es_source_health.get("cluster_name"),
                "url": SOURCE_ES_URL
            },
            "elasticsearch_dest": {
                "status": "connected",
                "cluster_name": es_dest_health.get("cluster_name"),
                "url": DEST_ES_URL
            },
            "model": model_status,
            "vector_search": "enabled" if app.state.model else "disabled"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "model": "unknown"
        }


# ==================== Run Instructions ====================
# To run this application:
# 1. Install dependencies: 
#    pip install fastapi uvicorn elasticsearch sentence-transformers
# 2. Ensure your model directory exists at MODEL_DIR path
# 3. Run: uvicorn main:app --reload
# 4. API docs available at: http://localhost:8000/docs
#
# Example requests:
# 
# Using size parameter:
# curl -X POST "http://localhost:8000/search" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "query": "aluminium",
#     "filter": {
#       "trade_type": "TRANSIT",
#       "in_vehicle_ids": ["avtomobil"]
#     },
#     "size": 10,
#     "use_vector": false
#   }'
#
# Using top_k parameter (same as size):
# curl -X POST "http://localhost:8000/search" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "query": "metal",
#     "filter": {
#       "trade_type": "IMPORT",
#       "in_vehicle_ids": ["deniz", "avtomobil"],
#       "out_vehicle_ids": ["avtomobil"]
#     },
#     "top_k": 10,
#     "alpha": 0.6,
#     "use_vector": true
#   }'
#
# Note: top_k is an alias for size. You can use either parameter.