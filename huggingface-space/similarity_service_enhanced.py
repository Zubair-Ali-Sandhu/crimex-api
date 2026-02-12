"""
CRIMEX Similarity Microservice - Enhanced Version
FastAPI endpoint for UC-18 Behavioral Pattern Analysis

Features:
- Search by existing case number
- Search with NEW case descriptions from website
- Real-time embedding generation
- CORS enabled for website integration

Deployment:
    pip install fastapi uvicorn sentence-transformers scikit-learn pandas numpy python-multipart
    uvicorn similarity_service_enhanced:app --host 0.0.0.0 --port 8001

API Documentation:
    http://localhost:8001/docs
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CRIMEX Similarity Engine",
    description="API for finding similar crime cases based on behavioral patterns",
    version="2.0.0"
)

# CORS configuration for website integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your website domain in production: ["https://yourwebsite.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded data
model = None
embeddings = None
df = None
case_index = None
geo_scaler = None
config = None

# Configuration class
class Config:
    """Configuration matching the notebook settings"""
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    WEIGHT_NLP = 0.60
    WEIGHT_CATEGORY = 0.15
    WEIGHT_GEO = 0.15
    WEIGHT_TEMPORAL = 0.10
    VALID_CATEGORIES = [
        'THEFT', 'ROBBERY', 'HARASSMENT', 'DOMESTIC_VIOLENCE',
        'FRAUD', 'BURGLARY', 'ASSAULT', 'OTHER'
    ]
    VALID_CITIES = [
        'Lahore', 'Faisalabad', 'Gujranwala', 'Multan', 'Sheikhupura', 'Sialkot'
    ]


@app.on_event("startup")
async def load_models():
    """Load models and data on startup"""
    global model, embeddings, df, case_index, geo_scaler, config

    try:
        logger.info("Loading models and data...")

        # Load configuration
        config = Config()

        # Load NLP model
        logger.info(f"Loading {config.EMBEDDING_MODEL}...")
        model = SentenceTransformer(config.EMBEDDING_MODEL)

        # Load pre-computed embeddings
        logger.info("Loading embeddings...")
        embeddings = np.load('description_embeddings.npy')

        # Load case data
        logger.info("Loading case data...")
        df = pd.read_csv('crime_reports_cleaned.csv')
        df['incident_date'] = pd.to_datetime(df['incident_date'])

        # Extract temporal features
        df['hour'] = df['incident_date'].dt.hour
        df['day_of_week'] = df['incident_date'].dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'].astype(np.float64) / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'].astype(np.float64) / 24.0)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'].astype(np.float64) / 7.0)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'].astype(np.float64) / 7.0)

        # Initialize geo scaler
        geo_scaler = MinMaxScaler()
        coords = df[['latitude', 'longitude']].values
        coords_normalized = geo_scaler.fit_transform(coords)
        df['lat_normalized'] = coords_normalized[:, 0]
        df['lng_normalized'] = coords_normalized[:, 1]

        # Build case index
        case_index = {cn: i for i, cn in enumerate(df['case_number'])}

        logger.info(f"✓ Loaded {len(df)} cases with {embeddings.shape[1]}-dim embeddings")
        logger.info(f"✓ Model ready: {config.EMBEDDING_MODEL}")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


# ============================================================================
# Request/Response Models
# ============================================================================

class LinkedCase(BaseModel):
    """Information about a linked case"""
    case_number: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    category: str
    city: str
    incident_date: str
    location: str
    snippet: str
    components: Optional[Dict[str, float]] = None


class SimilarityResponse(BaseModel):
    """Response with similar cases"""
    query_case: str
    total_results: int
    linked_cases: List[LinkedCase]
    processing_time_ms: float
    weights_used: Dict[str, float]


class ExistingCaseRequest(BaseModel):
    """Request to find cases similar to an existing case"""
    case_number: str = Field(..., example="CR-2025-000001")
    top_k: int = Field(default=5, ge=1, le=50)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    same_city_only: bool = Field(default=False)
    return_components: bool = Field(default=False)


class NewCaseRequest(BaseModel):
    """Request to find cases similar to a NEW case description"""
    description: str = Field(..., min_length=50, example="Suspect snatched mobile phone from victim near market...")
    category: str = Field(..., example="THEFT")
    city: str = Field(..., example="Lahore")
    latitude: float = Field(..., ge=29.5, le=34.0, example=31.5204)
    longitude: float = Field(..., ge=71.0, le=75.5, example=74.3587)
    incident_hour: Optional[int] = Field(default=None, ge=0, le=23)
    incident_day_of_week: Optional[int] = Field(default=None, ge=0, le=6)
    top_k: int = Field(default=5, ge=1, le=50)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    same_city_only: bool = Field(default=False)
    return_components: bool = Field(default=False)

    @validator('category')
    def validate_category(cls, v):
        if v not in Config.VALID_CATEGORIES:
            raise ValueError(f"Invalid category. Must be one of: {Config.VALID_CATEGORIES}")
        return v

    @validator('city')
    def validate_city(cls, v):
        if v not in Config.VALID_CITIES:
            raise ValueError(f"Invalid city. Must be one of: {Config.VALID_CITIES}")
        return v


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    cases_loaded: int
    model: str
    timestamp: str


# ============================================================================
# Similarity Computation Functions
# ============================================================================

def compute_temporal_features(hour: Optional[int], day_of_week: Optional[int]) -> np.ndarray:
    """Compute temporal features for a new case"""
    if hour is None:
        hour = 12  # Default to noon
    if day_of_week is None:
        day_of_week = 0  # Default to Monday

    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))
    dow_sin = float(np.sin(2 * np.pi * day_of_week / 7))
    dow_cos = float(np.cos(2 * np.pi * day_of_week / 7))

    return np.array([hour_sin, hour_cos, dow_sin, dow_cos], dtype=np.float64)


def compute_similarity_components(
    query_embedding: np.ndarray,
    query_category: str,
    query_coords_normalized: np.ndarray,
    query_temporal: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute all similarity components"""

    # Ensure all inputs are pure numpy float64 arrays (fixes numpy 2.x compat)
    query_embedding = np.asarray(query_embedding, dtype=np.float64)
    query_coords_normalized = np.asarray(query_coords_normalized, dtype=np.float64)
    query_temporal = np.asarray(query_temporal, dtype=np.float64)
    embeddings_f64 = np.asarray(embeddings, dtype=np.float64)

    # NLP similarity
    nlp_sim = cosine_similarity(query_embedding.reshape(1, -1), embeddings_f64).flatten()

    # Category similarity
    cat_sim = np.asarray((df['category'] == query_category).astype(float).values, dtype=np.float64)

    # Geographic similarity
    coords_db = np.asarray(df[['lat_normalized', 'lng_normalized']].values, dtype=np.float64)
    distances = np.linalg.norm(coords_db - query_coords_normalized, axis=1)
    geo_sim = np.exp(-distances * 5.0)

    # Temporal similarity
    temporal_db = np.asarray(df[['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']].values, dtype=np.float64)
    temp_distances = np.linalg.norm(temporal_db - query_temporal, axis=1)
    max_dist = float(np.sqrt(8.0))  # Convert to Python float to avoid ufunc issues
    temp_sim = 1.0 - (temp_distances / max_dist)

    return {
        'nlp': nlp_sim,
        'category': cat_sim,
        'geo': geo_sim,
        'temporal': temp_sim
    }


def combine_similarities(
    components: Dict[str, np.ndarray],
    weights: Dict[str, float]
) -> np.ndarray:
    """Combine similarity components with weights"""
    combined = (
        weights['nlp'] * components['nlp'] +
        weights['category'] * components['category'] +
        weights['geo'] * components['geo'] +
        weights['temporal'] * components['temporal']
    )
    return combined


def format_linked_cases(
    indices: np.ndarray,
    similarities: np.ndarray,
    components: Optional[Dict[str, np.ndarray]] = None
) -> List[LinkedCase]:
    """Format results as LinkedCase objects"""
    results = []

    for idx, sim in zip(indices, similarities):
        row = df.iloc[idx]

        result = LinkedCase(
            case_number=row['case_number'],
            similarity=float(sim),
            category=row['category'],
            city=row['city'],
            incident_date=str(row['incident_date']),
            location=row['location_address'][:100] + '...' if len(row['location_address']) > 100 else row['location_address'],
            snippet=row['description'][:200] + '...' if len(row['description']) > 200 else row['description']
        )

        if components:
            result.components = {
                'nlp': float(components['nlp'][idx]),
                'category': float(components['category'][idx]),
                'geo': float(components['geo'][idx]),
                'temporal': float(components['temporal'][idx])
            }

        results.append(result)

    return results


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/v1/similar-cases/existing", response_model=SimilarityResponse)
async def find_similar_to_existing(request: ExistingCaseRequest):
    """
    Find cases similar to an EXISTING case in the database.

    Use this endpoint when you have a case number and want to find related cases.
    """
    import time
    start = time.perf_counter()

    try:
        # Validate case exists
        if request.case_number not in case_index:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case number not found: {request.case_number}"
            )

        query_idx = case_index[request.case_number]
        query_row = df.iloc[query_idx]

        # Get query features
        query_embedding = embeddings[query_idx]
        query_category = query_row['category']
        query_coords = np.array([[query_row['lat_normalized'], query_row['lng_normalized']]])
        query_temporal = np.asarray(query_row[['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']].values, dtype=np.float64)

        # Compute similarities
        components = compute_similarity_components(
            query_embedding, query_category, query_coords, query_temporal
        )
        combined_sim = combine_similarities(components, {
            'nlp': config.WEIGHT_NLP,
            'category': config.WEIGHT_CATEGORY,
            'geo': config.WEIGHT_GEO,
            'temporal': config.WEIGHT_TEMPORAL
        })

        # Apply filters
        mask = np.ones(len(df), dtype=bool)
        mask[query_idx] = False  # Exclude self
        mask &= combined_sim >= request.min_similarity

        if request.same_city_only:
            mask &= df['city'] == query_row['city']

        # Get top-k
        valid_indices = np.where(mask)[0]
        valid_sims = combined_sim[mask]
        sorted_order = np.argsort(valid_sims)[::-1][:request.top_k]
        top_indices = valid_indices[sorted_order]
        top_sims = valid_sims[sorted_order]

        # Format results
        linked_cases = format_linked_cases(
            top_indices, top_sims,
            components if request.return_components else None
        )

        elapsed = (time.perf_counter() - start) * 1000

        return SimilarityResponse(
            query_case=request.case_number,
            total_results=len(linked_cases),
            linked_cases=linked_cases,
            processing_time_ms=elapsed,
            weights_used={
                'nlp': config.WEIGHT_NLP,
                'category': config.WEIGHT_CATEGORY,
                'geo': config.WEIGHT_GEO,
                'temporal': config.WEIGHT_TEMPORAL
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing existing case: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/v1/similar-cases/new", response_model=SimilarityResponse)
async def find_similar_to_new(request: NewCaseRequest):
    """
    Find cases similar to a NEW case description from your website.

    Use this endpoint when a user submits a new case and you want to find similar existing cases.
    This generates embeddings in real-time for the new description.
    """
    import time
    start = time.perf_counter()

    try:
        # Generate embedding for new description
        logger.info(f"Generating embedding for new {request.category} case in {request.city}")
        query_embedding = model.encode([request.description], normalize_embeddings=True)[0]

        # Normalize coordinates
        query_coords_raw = np.array([[request.latitude, request.longitude]])
        query_coords_normalized = geo_scaler.transform(query_coords_raw)[0]

        # Compute temporal features
        query_temporal = compute_temporal_features(request.incident_hour, request.incident_day_of_week)

        # Compute similarities
        components = compute_similarity_components(
            query_embedding, request.category, query_coords_normalized, query_temporal
        )
        combined_sim = combine_similarities(components, {
            'nlp': config.WEIGHT_NLP,
            'category': config.WEIGHT_CATEGORY,
            'geo': config.WEIGHT_GEO,
            'temporal': config.WEIGHT_TEMPORAL
        })

        # Apply filters
        mask = combined_sim >= request.min_similarity

        if request.same_city_only:
            mask &= df['city'] == request.city

        # Get top-k
        valid_indices = np.where(mask)[0]
        valid_sims = combined_sim[mask]
        sorted_order = np.argsort(valid_sims)[::-1][:request.top_k]
        top_indices = valid_indices[sorted_order]
        top_sims = valid_sims[sorted_order]

        # Format results
        linked_cases = format_linked_cases(
            top_indices, top_sims,
            components if request.return_components else None
        )

        elapsed = (time.perf_counter() - start) * 1000

        logger.info(f"Found {len(linked_cases)} similar cases in {elapsed:.2f}ms")

        return SimilarityResponse(
            query_case="NEW_CASE",
            total_results=len(linked_cases),
            linked_cases=linked_cases,
            processing_time_ms=elapsed,
            weights_used={
                'nlp': config.WEIGHT_NLP,
                'category': config.WEIGHT_CATEGORY,
                'geo': config.WEIGHT_GEO,
                'temporal': config.WEIGHT_TEMPORAL
            }
        )

    except Exception as e:
        logger.error(f"Error processing new case: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if df is not None else "not_ready",
        cases_loaded=len(df) if df is not None else 0,
        model=config.EMBEDDING_MODEL if config else "not_loaded",
        timestamp=datetime.now().isoformat()
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "CRIMEX Similarity Engine",
        "version": "2.0.0",
        "endpoints": {
            "existing_case": "/api/v1/similar-cases/existing",
            "new_case": "/api/v1/similar-cases/new",
            "health": "/health",
            "docs": "/docs"
        },
        "status": "operational"
    }
