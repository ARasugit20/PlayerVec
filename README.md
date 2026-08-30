# PlayerVec

[![CI](https://github.com/ARasugit20/PlayerVec/actions/workflows/ci.yml/badge.svg)](https://github.com/ARasugit20/PlayerVec/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ARasugit20/PlayerVec/graph/badge.svg)](https://codecov.io/gh/ARasugit20/PlayerVec)

A PyTorch autoencoder that compresses World Cup player stats into a 32-dim embedding space, then rolls them into **team style DNA** for **fixture matchup diagnosis**.

Three layers:
1. **PlayerVec core** — player similarity + UMAP clusters (no position labels)
2. **Team fingerprints** — minute-weighted squad style DNA (press, progression, aerial, finishing)
3. **Fixture briefs** — style clashes, structural gaps, adjustment cards, wildcard picks

```
FBref/StatsBomb → scraper → autoencoder → embeddings
  → FAISS (player search) + team fingerprints → fixture diagnosis → FastAPI → React UI
```

## Quick start

### Prerequisites

- Python 3.11+
- Node 20+ (for frontend)

### 1. Install & run pipeline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH=.

# Real World Cup data (FBref stats + StatsBomb roster metadata)
bash scripts/run_pipeline.sh

# Or synthetic sample data (CI only)
bash scripts/run_pipeline.sh sample
```

### 2. Start API

```bash
uvicorn api.main:app --reload --port 8000
```

### 3. Start frontend

```bash
cd client && npm install && npm run dev
```

Open http://localhost:5173

### Docker Compose

```bash
docker compose up --build
```

- API: http://localhost:8000
- UI: http://localhost:5173

## API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /players?q=Bell` | Autocomplete player names |
| `GET /similar?player=Erling+Haaland&k=5` | Top-k style matches |
| `GET /umap` | 2D UMAP coordinates for cluster map |
| `GET /teams` | List squads |
| `GET /team-fingerprint?team=Brazil` | Squad style DNA + archetype mix |
| `GET /fixture-brief?team_a=Brazil&team_b=France` | Matchup diagnosis report |
| `GET /tournament-outlook?team=Brazil` | Hardest stylistic matchups |

## Project structure

```
playervec/
├── data/           # scraper, features, parquet, embeddings
├── model/          # autoencoder, train, evaluate, embed
├── search/         # FAISS index, query, UMAP
├── team/           # fingerprints, fixture diagnosis, evaluation
├── api/            # FastAPI routes
├── client/         # React + Vite + Tailwind
├── docs/           # RESEARCH.md, INTERVIEW.md
└── scripts/        # run_pipeline.sh
```

## Architecture

- **Encoder:** 30 → 128 → 64 → 32 (BatchNorm, ReLU, Dropout)
- **Loss:** MSE reconstruction
- **Search:** FAISS IndexFlatIP (cosine similarity on L2-normalized embeddings)
- **Viz:** UMAP 2D projection

## Data sources

| Source | Use |
|--------|-----|
| `worldcup` (default) | FBref per-90 stats + StatsBomb jersey/nation/position |
| `statsbomb` | Event-aggregated stats from StatsBomb open data |
| `fbref` via soccerdata | FBref only; needs Chrome/Selenium |
| `--source sample` | Synthetic data for CI only |

## Research

See [docs/RESEARCH.md](docs/RESEARCH.md) for the ablation table (autoencoder vs PCA vs raw stats).

## Deployment

- **Backend:** Railway — deploy from `Dockerfile`, set `CORS_ORIGINS` to your Vercel URL
- **Frontend:** Vercel — set `VITE_API_URL` to your Railway API URL
- **CI:** GitHub Actions (`.github/workflows/ci.yml`)

## Interview prep

See [docs/INTERVIEW.md](docs/INTERVIEW.md) for talking points.
