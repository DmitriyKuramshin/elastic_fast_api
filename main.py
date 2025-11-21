from enum import Enum
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from elasticsearch import AsyncElasticsearch
from sentence_transformers import SentenceTransformer
import re
import asyncio
import logging

# ==================== Setup Logging ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    highlight: Optional[Dict[str, List[str]]] = None 

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
            highlight=hit.get("highlight"),
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
    use_highlight: bool = Field(default=False, description="Return Elasticsearch highlight snippets")
    
    def model_post_init(self, __context):
        if self.top_k is not None:
            self.size = self.top_k


# ==================== Query Builder ====================

def build_es_bool_query(req: SearchRequest) -> dict:
    f = req.filter

    query_text = req.query.strip()

    numeric_code = re.sub(r'\D', '', query_text)
    has_code = len(numeric_code) >= 6
    
    is_only_code = query_text.isdigit() and len(query_text) in [6, 8, 10]    

    # (field_name, boost)
    field_configs = [
        ("keywords_az_level1", 6),
        ("name_az_d4", 5),
        ("keywords_az_level2", 4),
        ("name_az_d3", 3),
        ("name_az_d2", 2),
        ("name_az_d1", 1),
    ]

    should_clauses = []
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
        for field, boost in field_configs:
            # match_phrase
            should_clauses.append({
                "match_phrase": {
                    field: {
                        "query": query_text,
                        "boost": boost,
                        "slop": 1,
                    }
                }
            })

            # match with fuzziness
            should_clauses.append({
                "match": {
                    field: {
                        "query": query_text,
                        "boost": boost,
                        "fuzziness": "AUTO",
                        "prefix_length": 2,
                    }
                }
            })

            # match_phrase_prefix
            should_clauses.append({
                "match_phrase_prefix": {
                    field: {
                        "query": query_text,
                        "boost": boost,
                        "max_expansions": 8,
                    }
                }
            })

            # span_near with fuzzy span_multi
            should_clauses.append({
                "span_near": {
                    "clauses": [
                        {
                            "span_multi": {
                                "match": {
                                    "fuzzy": {
                                        field: {
                                            "value": query_text,
                                            "fuzziness": "AUTO",
                                            "prefix_length": 2,
                                        }
                                    }
                                }
                            }
                        }
                    ],
                    "slop": 2,
                    "in_order": False,
                    "boost": boost,
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
    
    query = {"bool": bool_query}
    
    logger.info(f"Constructed BM25 bool query: {query}")
    
    return query

def build_hybrid_vector_query(req: SearchRequest, query_vector: List[float]) -> dict:
    """Build hybrid query combining BM25 and vector similarity with filters"""
    logger.info(f"Building hybrid vector query with alpha={req.alpha}")
    
    # Build the base BM25 query
    base_query = build_es_bool_query(req)
    bool_query = base_query["bool"]

    # Parent weights
    parent_coefficient = 0.3
    parent_fields = {
        "embedding_d1": 1.0 * parent_coefficient,
        "embedding_d2": 1.5 * parent_coefficient,
        "embedding_d3": 2.0 * parent_coefficient,
        "embedding_d4": 1.0  # main leaf, full weight
    }
    
    logger.info(f"Embedding weights: d1={parent_fields['embedding_d1']}, "
                f"d2={parent_fields['embedding_d2']}, d3={parent_fields['embedding_d3']}, "
                f"d4={parent_fields['embedding_d4']}")
    
    # Build script_score query for hybrid search with null checks
    query = {
        "script_score": {
            "query": {"bool": bool_query},
            "script": {
                "source": """
                double cosine = 0.0;
                
                // Level embeddings with null/empty checks
                if (doc.containsKey('embedding_d1') && doc['embedding_d1'].size() > 0) {
                    cosine += params.alpha_level1 * (cosineSimilarity(params.query_vector, 'embedding_d1') + 1.0)/2.0;
                }
                if (doc.containsKey('embedding_d2') && doc['embedding_d2'].size() > 0) {
                    cosine += params.alpha_level2 * (cosineSimilarity(params.query_vector, 'embedding_d2') + 1.0)/2.0;
                }
                if (doc.containsKey('embedding_d3') && doc['embedding_d3'].size() > 0) {
                    cosine += params.alpha_level3 * (cosineSimilarity(params.query_vector, 'embedding_d3') + 1.0)/2.0;
                }
                if (doc.containsKey('embedding_d4') && doc['embedding_d4'].size() > 0) {
                    cosine += params.alpha_level4 * (cosineSimilarity(params.query_vector, 'embedding_d4') + 1.0)/2.0;
                }
                
                // BM25 score normalization
                double bm25 = _score / (_score + 10.0);
                
                // Hybrid score calculation
                return params.alpha * cosine + (1 - params.alpha) * bm25;
                """,
                "params": {
                    "query_vector": query_vector,
                    "alpha": req.alpha,
                    "alpha_level1": parent_fields["embedding_d1"],
                    "alpha_level2": parent_fields["embedding_d2"],
                    "alpha_level3": parent_fields["embedding_d3"],
                    "alpha_level4": parent_fields["embedding_d4"],
                }
            }
        }
    }
    
    logger.info(f"Built script_score query with vector dimension: {len(query_vector)}")
    
    return query

def build_highlight_config() -> dict:
    return {
        "pre_tags": ["<mark>"],
        "post_tags": ["</mark>"],
        "fields": {
            "name_az_d1": {},
            "name_az_d2": {},
            "name_az_d3": {},
            "name_az_d4": {},
            "keywords_az_level1": {},
            "keywords_az_level2": {},
        },
    }


# ==================== FastAPI Application ====================

# Since both source and dest are the same, we only need one connection
ES_URL = "http://10.3.3.16:9200"
ES_API_KEY = ""
ES_INDEX = "flattened_hscodes_v4_copy"

MODEL_DIR = "m12_1e"

app = FastAPI(
    title="Hybrid Search API with Vector Search",
    version="2.0.0",
    description="FastAPI service for hybrid search with Elasticsearch and vector embeddings"
)


def load_model():
    """Load the sentence transformer model"""
    logger.info(f"Loading model from: {MODEL_DIR}")
    try:
        model = SentenceTransformer(MODEL_DIR)
        logger.info(f"✅ Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as e:
        logger.warning(f"⚠️ Warning: Could not load model from {MODEL_DIR}: {e}")
        logger.warning("Vector search will be disabled.")
        return None


@app.on_event("startup")
async def startup():
    """Initialize Elasticsearch connection and load model on startup"""
    app.state.es = AsyncElasticsearch(hosts=[ES_URL], api_key=ES_API_KEY)
    logger.info(f"✅ Connected to Elasticsearch at {ES_URL}")
    
    # Load the embedding model
    app.state.model = load_model()


@app.on_event("shutdown")
async def shutdown():
    """Close Elasticsearch connection on shutdown"""
    await app.state.es.close()
    logger.info("Elasticsearch connection closed")


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
    
    logger.info(f"Search request: query='{req.query}', size={req.size}, "
                f"use_vector={req.use_vector}, alpha={req.alpha}")
    
    # Check if vector search is enabled and model is available
    use_vector_search = req.use_vector and app.state.model is not None
    
    if req.use_vector and app.state.model is None:
        logger.warning("Vector search requested but model is not loaded. Falling back to BM25.")
    
    highlight_conf = build_highlight_config() if req.use_highlight else None

    if use_vector_search:
        logger.info("Using HYBRID search mode (BM25 + Vector)")
        
        # Generate query embedding (run in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        query_vector = await loop.run_in_executor(
            None, 
            lambda: app.state.model.encode(req.query).tolist()
        )
        
        logger.info(f"Generated query embedding with {len(query_vector)} dimensions")
        
        # Build hybrid query with vector similarity
        es_query = build_hybrid_vector_query(req, query_vector)
        
        try:
            search_kwargs = {
                "index": ES_INDEX,
                "query": es_query,
                "size": req.size,
            }
            if highlight_conf:
                search_kwargs["highlight"] = highlight_conf
                logger.info("Highlighting enabled")

            logger.info(f"Executing hybrid search on index: {ES_INDEX}")
            resp = await app.state.es.search(**search_kwargs)
            logger.info(f"Hybrid search completed successfully")
            
        except Exception as e:
            logger.error(f"Elasticsearch hybrid search error: {e}", exc_info=True)
            raise HTTPException(
                status_code=502,
                detail=f"Elasticsearch (hybrid) error: {e}"
            )
        
        hits_data = (resp or {}).get("hits", {})
        hits = hits_data.get("hits", []) or []
        total_hits = hits_data.get("total", {})
        
        logger.info(f"Retrieved {len(hits)} hits from hybrid search")
        
    else:
        logger.info("Using BM25-only search mode")
        
        # Build standard BM25 query
        es_query = build_es_bool_query(req)
        
        try:
            search_kwargs = {
                "index": ES_INDEX,
                "query": es_query,
                "size": req.size,
            }
            if highlight_conf:
                search_kwargs["highlight"] = highlight_conf
                logger.info("Highlighting enabled")
            
            logger.info(f"Executing BM25 search on index: {ES_INDEX}")
            resp = await app.state.es.search(**search_kwargs)
            logger.info(f"BM25 search completed successfully")
            
        except Exception as e:
            logger.error(f"Elasticsearch BM25 search error: {e}", exc_info=True)
            raise HTTPException(
                status_code=502,
                detail=f"Elasticsearch (BM25) error: {e}"
            )
        
        # Parse hits and total count
        hits_data = (resp or {}).get("hits", {})
        hits = hits_data.get("hits", []) or []
        total_hits = hits_data.get("total", {})
        
        logger.info(f"Retrieved {len(hits)} hits from BM25 search")
    
    # Extract total count (Elasticsearch can return this in different formats)
    if isinstance(total_hits, dict):
        total_count = total_hits.get("value", 0)
    else:
        total_count = total_hits or 0
    
    logger.info(f"Total hits available: {total_count}")
    
    ranked = [ElasticDocument.from_es_hit(h) for h in hits]
    
    # Log top 3 results
    if ranked:
        logger.info(f"Top result: code={ranked[0].code}, score={ranked[0].score:.4f}")
        if len(ranked) > 1:
            logger.info(f"2nd result: code={ranked[1].code}, score={ranked[1].score:.4f}")
        if len(ranked) > 2:
            logger.info(f"3rd result: code={ranked[2].code}, score={ranked[2].score:.4f}")
    
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
    """Check if the service, Elasticsearch connection, and model are healthy"""
    try:
        es_health = await app.state.es.info()
        model_status = "loaded" if app.state.model is not None else "not loaded"
        
        return {
            "status": "healthy",
            "elasticsearch": {
                "status": "connected",
                "cluster_name": es_health.get("cluster_name"),
                "url": ES_URL,
                "index": ES_INDEX
            },
            "model": {
                "status": model_status,
                "path": MODEL_DIR
            },
            "vector_search": "enabled" if app.state.model else "disabled"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "model": "unknown"
        }