import os
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data

os.makedirs("graphs", exist_ok=True)

# SBERT model (fast + good)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_graph(sample):
    claim = sample["claim"]
    evidence_list = sample["evidence"]
    y = int(sample["label"])

    texts = [claim] + evidence_list

    # node embeddings
    x = embedder.encode(texts, convert_to_tensor=True).float()

    num_nodes = x.size(0)

    # edges: claim <-> each evidence
    src = []
    dst = []
    for i in range(1, num_nodes):
        src += [0, i]
        dst += [i, 0]

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long)
    )
    return data

def convert_jsonl_to_graphs(input_jsonl, output_pt, limit=None):
    graphs = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc=f"Building graphs from {input_jsonl}")):
            if limit is not None and idx >= limit:
                break
            sample = json.loads(line)
            graphs.append(build_graph(sample))

    torch.save(graphs, output_pt)
    print(f"Saved graphs -> {output_pt} | total = {len(graphs)}")

if __name__ == "__main__":
    convert_jsonl_to_graphs("data/train_clean.jsonl", "graphs/train_graphs.pt")
    convert_jsonl_to_graphs("data/val_clean.jsonl", "graphs/val_graphs.pt")
