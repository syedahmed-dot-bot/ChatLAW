from typing import List, Dict
from pypdf import PdfReader

def pdf_to_text_blocks(path: str)-> List[Dict]:
    """Return [{'page': int, 'text: str}] for each page"""
    reader = PdfReader(path)
    blocks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            blocks.append({"page": i+ 1, "text": " ".join(text.split())})
    
    return blocks

def sliding_window(words: List[str], size: int, overlap: int) -> List[str]:
    step = max(size - overlap, 1)
    out = []
    for i in range(0, len(words), step):
        chunk = words[i: 1 + size]
        if chunk:
            out.append(" ".join(chunk))
    
    return out

def chunk_blocks(blocks: List[Dict], size: int = 900, overlap: int = 120) -> List[Dict]:
    """chunk page-level blocks into token-sized segments (word-based)"""
    chunks: List[Dict] = []
    for b in blocks: 
        words = b["text"].split()
        segs = sliding_window(words, size = size, overlap = overlap)
        for j, seg in enumerate(segs):
            chunks.append({
                "text": seg,
                "metadata": {"page": b["page"], "chunk_id": j}
            })
        
    return chunks
