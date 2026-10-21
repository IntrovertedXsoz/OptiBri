"""Build tokenized calibration/eval sequences: Wikipedia (wikitext-103) + Python source code."""
import os, glob, random, urllib.request, json, numpy as np
from tokenizers import Tokenizer
CACHE = os.environ.get("GLM_CACHE", os.path.expanduser("~/.glm_cache"))
def _dl(url, name):
    p = os.path.join(CACHE, name)
    if not os.path.exists(p): open(p, "wb").write(urllib.request.urlopen(url).read())
    return p
def tokenizer():
    return Tokenizer.from_file(_dl("https://huggingface.co/zai-org/GLM-5.2/resolve/main/tokenizer.json", "tokenizer.json"))
def build(n_wiki=6, n_code=2, T=1024, seed=0):
    import pyarrow.parquet as pq
    tok = tokenizer(); rng = random.Random(seed)
    wiki = pq.read_table(_dl("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/test-00000-of-00001.parquet", "wt103_test.parquet")).column("text").to_pylist()
    wiki_ids = tok.encode("".join(wiki)).ids
    code = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob("/usr/lib/python3*/[a-z]*.py"))[:60])
    code_ids = tok.encode(code).ids
    seqs = []
    for src, n in ((wiki_ids, n_wiki), (code_ids, n_code)):
        for _ in range(n):
            s = rng.randrange(0, len(src) - T); seqs.append(src[s:s + T])
    return np.array(seqs, dtype=np.int64)
if __name__ == "__main__":
    s = build(); np.save(os.path.join(CACHE, "calib_8x1024.npy"), s)
    print(s.shape, tokenizer().decode(s[0][:40].tolist())[:200].replace("\n", " "), "||", tokenizer().decode(s[-1][:30].tolist())[:120].replace("\n", " "))
