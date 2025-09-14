from pathlib import Path
import json
from collections import defaultdict, Counter

INDEX_DIR = Path("data/index")
texts = json.loads((INDEX_DIR / "texts.json").read_text())
meta  = json.loads((INDEX_DIR / "meta.json").read_text())

by_source = defaultdict(list)
for i, m in enumerate(meta):
    by_source[m["source"]].append((i, m))

print(f"Total chunks: {len(texts)}")
print(f"Total sources (PDFs): {len(by_source)}\n")

for src, items in sorted(by_source.items(), key=lambda x: len(x[1]), reverse=True):
    # page is flattened: use m.get("page")
    pages = Counter(m.get("page") for _, m in items if m.get("page") is not None)
    print(f"{src}: {len(items)} chunks | ~{len(pages) + 1} unique pages seen")
