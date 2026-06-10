# PlayerVec — Interview Talking Points

## Elevator pitch (30 seconds)

PlayerVec is an unsupervised representation learning system that compresses World Cup player stats into 32-dimensional style embeddings using a PyTorch autoencoder. I built a full product on top: FAISS similarity search, a FastAPI backend, and a React UI with an interactive UMAP cluster map. The model never sees position labels during training, yet attackers, midfielders, and defenders self-organize in embedding space.

## Technical depth

### Why an autoencoder?

- Tabular sports stats are structured and interpretable — an autoencoder is debuggable (reconstruction loss, feature ablation) unlike a black-box transformer.
- 30 → 32 compression forces the network to learn a compact "playing style" manifold.
- MSE reconstruction loss ensures embeddings retain stat information needed for meaningful similarity.

### Architecture choices

- **Encoder:** 30 → 128 → 64 → 32 with BatchNorm, ReLU, Dropout(0.2)
- **Decoder:** symmetric reconstruction path
- **Embeddings:** L2-normalized for cosine similarity via FAISS IndexFlatIP
- **Baselines:** PCA (32 components) and raw standardized stats — compared with silhouette score by position

### Data pipeline

- **Training:** StatsBomb open data + historical FBref World Cups (2014–2022)
- **Inference:** Live 2026 FBref stats via `soccerdata` as the tournament runs
- ~30 per-90 features: goals, xG, pressures, progressive passes, aerials, etc.

### Product decisions

- `/similar?player=X&k=5` — top-k cosine neighbors
- `/umap` — 2D projection for the cluster map
- Click a dot → search that player's style neighborhood

## Questions I can answer well

1. **How do you evaluate unsupervised embeddings?** Silhouette by position (proxy), reconstruction loss, qualitative neighbor inspection, ablation vs PCA.
2. **Why not just use PCA?** Honest answer: PCA is a strong baseline on tabular data. The autoencoder wins when it captures nonlinear interactions (e.g., high pressures + high progressive passes = specific midfielder archetype).
3. **How would you improve it?** Contrastive loss with match co-occurrence pairs, minutes-weighted training, season-aware normalization, position-agnostic style labels from scouting reports.
4. **Deployment:** Docker Compose locally, Railway (API) + Vercel (frontend), GitHub Actions CI.

## Metrics to cite

- Reconstruction val MSE (from `model/checkpoints/history.json`)
- Silhouette scores (from `docs/RESEARCH.md` after running evaluate)
- ~400 players in sample mode; scales to full World Cup squads with live scraper

## What makes this different from a class project

- Real tournament data (not MNIST)
- Unsupervised learning (harder to evaluate, more interesting)
- Full-stack product a non-technical user can demo
- Research artifact (ablation table) showing intellectual honesty
