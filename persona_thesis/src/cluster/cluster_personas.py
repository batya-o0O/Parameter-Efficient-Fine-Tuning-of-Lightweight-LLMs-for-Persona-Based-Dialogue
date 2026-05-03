import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from src.utils.io import load_json, save_json, ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default="data/personas_rolebench.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    personas = load_json(args.personas)
    texts = [p["description"] for p in personas]
    ids = [p["persona_id"] for p in personas]

    embedder = SentenceTransformer(args.embed_model)
    X = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    km = KMeans(n_clusters=args.k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    clusters = {}
    for pid, desc, lab in zip(ids, texts, labels):
        clusters.setdefault(str(lab), []).append({"persona_id": pid, "description": desc})

    out = args.out or f"data/clusters_k{args.k}.json"
    ensure_dir("data")
    save_json(out, {"k": args.k, "embed_model": args.embed_model, "clusters": clusters})
    print("Saved", out)

if __name__ == "__main__":
    main()