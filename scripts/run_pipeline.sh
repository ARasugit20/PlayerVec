#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE="${1:-sample}"

echo "==> Scraping data (source=$SOURCE)"
python -m data.scraper --source "$SOURCE"

echo "==> Training autoencoder"
python -m model.train --epochs 100

echo "==> Generating embeddings"
python -m model.embed

echo "==> Building FAISS index"
python -m search.index

echo "==> Computing UMAP projection"
python -m search.umap_project

echo "==> Running ablation study"
python -m model.evaluate

echo "==> Pipeline complete"
