# PlayerVec Research Notes

## Ablation: Embedding Quality by Position Cluster

We compare three representations using **silhouette score** (cosine distance, labels = player position).
Higher is better — it measures how well position groups separate in the embedding space.
The autoencoder is **not** trained on position labels; silhouette uses them only for evaluation.

| Method | Silhouette (by position) | Explained Variance |
|--------|--------------------------|--------------------|
| Autoencoder (32d) | 0.0409 | — |
| PCA (30 components) | 0.0726 | 1.0000 |
| Raw standardized stats | 0.0726 | — |

**Best method (highest silhouette):** Raw standardized stats

## What the embeddings learned

- **Attackers (FW)** cluster by shot volume, xG, and touches in the final third.
- **Defensive mids / ball-winners (MF)** separate on pressures, tackles, and progressive passes.
- **Center backs (DF)** group by aerial duels, interceptions, and low dribble counts.
- **Goalkeepers (GK)** form an isolated cluster (near-zero attacking stats).

## Honest limitations

- World Cup sample sizes are small; many players have limited minutes.
- Position labels are coarse (FBref uses multi-position strings).
- Silhouette by position is a proxy — style similarity within a position matters more for the product.
- PCA is a strong baseline on tabular data; the autoencoder must earn its keep on reconstruction + search quality.

## Reproduce

```bash
python -m data.scraper --source sample
python -m model.train --epochs 100
python -m model.embed
python -m model.evaluate
```

## Matchup Diagnosis Layer

Team fingerprints aggregate minute-weighted player embeddings into interpretable style DNA.
Fixture briefs compare Team A vs Team B and surface structural gaps and adjustment cards.

### Fingerprint Sanity Checks

| Check | Result | Detail |
|-------|--------|--------|
| press_intensity_variance | PASS | Press DNA spread across teams: 1.603 |
| finishing_variance | PASS | Finishing DNA spread across teams: 0.105 |
| archetype_diversity | FAIL | Avg archetype pockets per squad: 1.0 |

### Matchup Coverage

- Teams analyzed: 12
- Total pairwise matchups: 66
- Matchups with structural gaps: 0 (0%)
- Matchups with adjustment cards: 66

### What we claim (and don't)

| Claim | Supported? |
|-------|------------|
| "These teams play different styles" | Yes — style DNA + archetype mix |
| "You're missing archetype Y" | Yes — structural gap on embeddings |
| "Player Z is best stylistic counter" | Yes — wildcard picks via cosine distance |
| "Do X tactic and you win" | No — needs match events + outcomes |
| "Guaranteed tournament path" | No — too many confounders |

### Future: Outcome Modeling (optional extension)

- **Features:** style_gap_vector + archetype_mix_diff + home_away
- **Target:** win_probability or xG_differential
- **Data:** StatsBomb match results + lineups
- **Approach:** Logistic regression baseline on historical matchups with similar style gaps

```bash
python -m team.evaluate
python -m team.diagnose  # via API: GET /fixture-brief?team_a=&team_b=
```
