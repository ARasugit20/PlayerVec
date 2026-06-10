# PlayerVec — Interview Talking Points

## Elevator pitch (30 seconds)

PlayerVec is an unsupervised representation learning system that compresses World Cup player stats into 32-dimensional style embeddings, then rolls them into team style DNA for matchup diagnosis. I built a full product: FAISS player search, team fingerprinting, fixture briefs with gap analysis and adjustment cards, FastAPI backend, and React UI. The model never sees position labels during training, yet attackers, midfielders, and defenders self-organize — and team-level gaps surface as scouting hypotheses, not black-box predictions.

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

- **Layer 1:** `/similar?player=X&k=5` — player style neighbors
- **Layer 2:** `/team-fingerprint?team=` — minute-weighted squad style DNA
- **Layer 3:** `/fixture-brief?team_a=&team_b=` — style clashes, structural gaps, adjustment cards
- `/tournament-outlook?team=` — hardest stylistic matchups in bracket
- `/umap` — 2D cluster map for player-level exploration

### Matchup diagnosis (the differentiator)

- Team fingerprints = minute-weighted embedding centroids + interpretable style dimensions (press, progression, aerial, finishing)
- Archetype mix via KMeans on embeddings (high_presser, deep_progressor, finisher, etc.)
- Fixture brief compares DNA, finds cosine-distance structural gaps, suggests wildcard lineup picks
- **Honest framing:** scouting memo language, not "do X and you win" — outcome modeling is documented future work

## Questions I can answer well

1. **How do you evaluate unsupervised embeddings?** Silhouette by position (proxy), reconstruction loss, ablation vs PCA, fingerprint sanity checks, matchup gap coverage rate.
2. **Why not just use PCA?** Honest answer: PCA is a strong baseline on tabular data. The autoencoder wins when it captures nonlinear interactions (e.g., high pressures + high progressive passes = specific midfielder archetype).
3. **How do you evaluate matchup diagnosis without outcomes?** Structural gap rate across pairwise matchups, qualitative sanity (press/finishing variance across teams), explicit limits on tactical certainty.
4. **How would you improve it?** Supervised outcome model on style-gap features + StatsBomb match results; contrastive loss with co-occurrence pairs; live re-embedding after each round.
5. **Deployment:** Docker Compose locally, Railway (API) + Vercel (frontend), GitHub Actions CI.

## Metrics to cite

- Reconstruction val MSE (from `model/checkpoints/history.json`)
- Silhouette scores (from `docs/RESEARCH.md` after running evaluate)
- ~400 players in sample mode; scales to full World Cup squads with live scraper

## What makes this different from a class project

- Real tournament data (not MNIST)
- Unsupervised learning (harder to evaluate, more interesting)
- Full-stack product a non-technical user can demo
- Research artifact (ablation table) showing intellectual honesty
