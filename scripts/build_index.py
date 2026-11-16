"""
Download the medical corpora and build the retrieval indexes.

    # 1. download (textbooks ~0.2 GB from HuggingFace; statpearls ~2 GB from NCBI)
    python scripts/build_index.py download --corpora textbooks statpearls

    # 2. build BM25 + MedCPT indexes over them (GPU: ~15 min for ~427k snippets)
    python scripts/build_index.py build --corpora textbooks statpearls

    # smoke test on a laptop: one textbook, first 2,000 snippets, CPU
    python scripts/build_index.py build --corpora textbooks --files First_Aid_Step2 --limit 2000

Everything lands in $DATA_DIR (see src/clinical_agent/config.py).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clinical_agent.config import get_settings  # noqa: E402
from src.clinical_agent.retrieval.corpus import (CorpusStore, download_statpearls,  # noqa: E402
                                                 download_textbooks, read_chunk_dir)
from src.clinical_agent.retrieval.indexes import BM25Index, DenseIndex, MedCPTEncoder  # noqa: E402


def download(corpora, root):
    for name in corpora:
        print(f"Downloading {name} into {root} ...")
        if name == "textbooks":
            download_textbooks(root)
        elif name == "statpearls":
            download_statpearls(root)
        else:
            raise ValueError(f"Unknown corpus: {name}")


def build(corpora, root, index_dir, files=None, limit=None, device=None, batch_size=128):
    snippets = []
    for name in corpora:
        chunk_dir = os.path.join(root, name, "chunk")
        for snippet in read_chunk_dir(chunk_dir, name):
            # --files keeps only snippets from some books (for a quick test on a laptop)
            if files:
                wanted = False
                for prefix in files:
                    if snippet.id.startswith(prefix):
                        wanted = True
                if not wanted:
                    continue
            snippets.append(snippet)
    if limit:
        snippets = snippets[:limit]
    corpus = CorpusStore(snippets)
    print(f"Corpus: {len(corpus):,} snippets from {', '.join(corpora)}")

    os.makedirs(index_dir, exist_ok=True)
    corpus.save(os.path.join(index_dir, "corpus.jsonl"))

    start = time.time()
    BM25Index().build(corpus.texts()).save(os.path.join(index_dir, "bm25"))
    print(f"BM25 index built in {time.time() - start:.0f}s")

    start = time.time()
    dense = DenseIndex(encoder=MedCPTEncoder(device=device)).build(corpus.titles, corpus.contents,
                                                                    batch_size=batch_size)
    dense.save(os.path.join(index_dir, "medcpt"))
    print(f"MedCPT index built in {time.time() - start:.0f}s -> {index_dir}")


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["download", "build"])
    parser.add_argument("--corpora", nargs="+", default=["textbooks", "statpearls"])
    parser.add_argument("--files", nargs="*", help="only snippets whose id starts with these (smoke tests)")
    parser.add_argument("--limit", type=int, help="only the first N snippets (smoke tests)")
    parser.add_argument("--index-dir", default=settings.index_dir)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if args.step == "build" and args.device is None:
        # Fail loudly instead of silently encoding ~427k passages on CPU (many hours).
        # This happened once: a torch build newer than the GPU driver reports no GPU.
        import torch
        if not torch.cuda.is_available():
            sys.exit("No GPU visible to torch (driver too old for this torch build?).\n"
                     "Check: python -c \"import torch; print(torch.version.cuda, torch.cuda.is_available())\"\n"
                     "Fix torch (see docs/RUNNING_BENCHMARKS.md), or pass --device cpu on purpose.")

    if args.step == "download":
        download(args.corpora, settings.corpus_dir)
    else:
        build(args.corpora, settings.corpus_dir, args.index_dir, args.files, args.limit, args.device, args.batch_size)


if __name__ == "__main__":
    main()
