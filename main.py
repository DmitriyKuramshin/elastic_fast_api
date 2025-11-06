from enum import Enum
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from elasticsearch import AsyncElasticsearch
import re

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
    trading_types: List[TradingType] = Field(default_factory=list)
    in_vehicle_ids: List[VehicleType] = Field(default_factory=list)
    out_vehicle_ids: List[VehicleType] = Field(default_factory=list)


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
    size: int = Field(default=10, ge=1, le=200)


# ==================== Query Builder ====================

def build_es_bool_query(req: SearchRequest) -> dict:
    f = req.filter

    query_text = req.query.strip()
    
    numeric_code = re.sub(r'\D', '', query_text)
    has_code = len(numeric_code) >= 6
    
    # Check if query is ONLY a code (no other characters except digits)
    is_only_code = query_text.isdigit() and len(query_text) in [6, 8, 10]
    
    should_clauses = []
    
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
        # Parent fields with slight boosting
        parent_fields = [
            ("name_az_d1", 0.15),
            ("name_az_d2", 0.15),
            ("name_az_d3", 0.15),
        ]
        
        for field, boost in parent_fields:
            should_clauses.append({
                "match": {
                    field: {
                        "query": req.query,
                        "boost": boost,
                        "fuzziness": "AUTO"
                    }
                }
            })
            should_clauses.append({
                "prefix": {
                    field: {
                        "value": req.query,
                        "boost": boost
                    }
                }
            })
        
        # name_az_d4 with higher boosting
        should_clauses.append({
            "match": {
                "name_az_d4": {
                    "query": req.query,
                    "boost": 1.0,
                    "fuzziness": "AUTO"
                }
            }
        })
        
        should_clauses.append({
            "prefix": {
                "name_az_d4": {
                    "value": req.query,
                    "boost": 2.0
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
    
    bool_query = {
        "should": should_clauses,
        "minimum_should_match": 1
    }
    
    filters = []
    
    # Apply trading_types filter on nested field
    if f.trading_types:
        filters.append({
            "nested": {
                "path": "tradings",
                "query": {
                    "terms": {"tradings.tradeType": [tt.value for tt in f.trading_types]}
                }
            }
        })
    
    # Apply inVehicleId filter on nested field
    if f.in_vehicle_ids:
        filters.append({
            "nested": {
                "path": "tradings",
                "query": {
                    "terms": {"tradings.inVehicleId": [v.to_id() for v in f.in_vehicle_ids]}
                }
            }
        })
    
    # Apply outVehicleId filter on nested field
    if f.out_vehicle_ids:
        filters.append({
            "nested": {
                "path": "tradings",
                "query": {
                    "terms": {"tradings.outVehicleId": [v.to_id() for v in f.out_vehicle_ids]}
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


# ==================== FastAPI Application ====================

ES_URL = "https://c70506d900e44618bd38984c5803a2ea.us-central1.gcp.cloud.es.io:443"
ES_API_KEY = "VDVCdlZab0J4Q0dCdlkwSTJEOEg6aFNud3BhSkN0QWxRR1dmVVpCdkotdw=="
ES_INDEX = "flattened_hscodes"

app = FastAPI(
    title="Hybrid Search API",
    version="1.0.0",
    description="FastAPI service for hybrid search with Elasticsearch"
)


@app.on_event("startup")
async def startup():
    """Initialize Elasticsearch connection on startup"""
    app.state.es = AsyncElasticsearch(hosts=[ES_URL], api_key=ES_API_KEY)
    print(f"Connected to Elasticsearch at {ES_URL}")


@app.on_event("shutdown")
async def shutdown():
    """Close Elasticsearch connection on shutdown"""
    await app.state.es.close()
    print("Elasticsearch connection closed")


@app.post(
    "/search",
    response_model=HybridRetrievedResponseSet,
    response_model_by_alias=True,
    summary="Hybrid Search",
    description="Perform hybrid search across products and organizations"
)
async def search(req: SearchRequest) -> HybridRetrievedResponseSet:
    """
    Execute a hybrid search query against Elasticsearch.
    
    Args:
        req: SearchRequest containing query text, filters, and size
        
    Returns:
        HybridRetrievedResponseSet with ranked search results
    """
    # Build Elasticsearch query
    es_query = build_es_bool_query(req)
    
    try:
        # Execute search
        resp = await app.state.es.search(
            index=ES_INDEX,
            query=es_query,
            size=req.size
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Elasticsearch error: {e}"
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
    """Check if the service and Elasticsearch are healthy"""
    try:
        es_health = await app.state.es.info()
        return {
            "status": "healthy",
            "elasticsearch": "connected",
            "cluster_name": es_health.get("cluster_name")
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "elasticsearch": "disconnected",
            "error": str(e)
        }


# ==================== Run Instructions ====================
# To run this application:
# 1. Install dependencies: pip install fastapi uvicorn elasticsearch
# 2. Set environment variables (optional):
#    export ES_URL="http://localhost:9200"
#    export ES_INDEX="hybrid-index"
# 3. Run: uvicorn main:app --reload
# 4. API docs available at: http://localhost:8000/docs