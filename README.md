---
title: CRIMEX Pattern Analysis API
emoji: 🚔
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# CRIMEX Crime Pattern Analysis API

ML-powered crime pattern detection system using behavioral analysis for Punjab Police.

## API Endpoints

- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)
- `POST /api/v1/similar-cases/new` - Find similar cases for new incidents
- `POST /api/v1/similar-cases/existing` - Find similar cases by case number

## Usage

1. Wait for the Space to build (5-10 minutes first time)
2. Access the interactive API documentation at `/docs`
3. Try the `/health` endpoint to verify it's running
4. Use `/api/v1/similar-cases/new` to find similar crime cases

## Example Request

```json
POST /api/v1/similar-cases/new
{
  "description": "Suspect snatched mobile phone from victim near market on motorcycle",
  "category": "THEFT",
  "city": "Lahore",
  "latitude": 31.5204,
  "longitude": 74.3587,
  "top_k": 5
}
```

## Technology Stack

- **Model**: SentenceTransformer (all-MiniLM-L6-v2)
- **Framework**: FastAPI + Uvicorn
- **Features**: Multi-modal similarity analysis
  - NLP-based description matching (60%)
  - Geographic proximity (15%)
  - Category matching (15%)
  - Temporal patterns (10%)

## Dataset

- 5,000+ crime reports from 6 cities in Punjab
- Pre-computed embeddings for fast similarity search
- Categories: THEFT, ROBBERY, FRAUD, BURGLARY, ASSAULT, MURDER

## Performance

- Response time: <100ms for similarity search
- Supports 5,000+ cases in database
- Real-time inference with pre-computed embeddings
