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
