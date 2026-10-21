"""Fetch individual tensors from the GLM-5.2 safetensors shards on Hugging Face
using HTTP range requests, so we never need to download whole shards."""
import json, struct, urllib.request, os, concurrent.futures as cf
import numpy as np

REPO = os.environ.get("GLM_REPO", "zai-org/GLM-5.2")
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
CACHE = os.environ.get("GLM_CACHE", os.path.expanduser("~/.glm_cache"))
os.makedirs(CACHE, exist_ok=True)
_hdr = {}

def _get(url, a, b):
    r = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    for i in range(5):
        try:
            return urllib.request.urlopen(r, timeout=120).read()
        except Exception:
            if i == 4: raise

def weight_map():
    p = os.path.join(CACHE, "index.json")
    if not os.path.exists(p):
        open(p, "wb").write(urllib.request.urlopen(BASE + "model.safetensors.index.json").read())
    return json.load(open(p))["weight_map"]

def header(shard):
    if shard not in _hdr:
        p = os.path.join(CACHE, shard + ".hdr.json")
        if os.path.exists(p):
            _hdr[shard] = json.load(open(p))
        else:
            n = struct.unpack("<Q", _get(BASE + shard, 0, 7))[0]
            h = json.loads(_get(BASE + shard, 8, 7 + n)); h["__off__"] = 8 + n
            json.dump(h, open(p, "w")); _hdr[shard] = h
    return _hdr[shard]

def bf16_to_f32(raw):
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)

def fetch(name, wm=None, dtype=np.float32, cache=True):
    """Return tensor `name` as a numpy array (bf16 -> float32). With cache=True the raw bytes are kept on disk."""
    wm = wm or weight_map()
    p = os.path.join(CACHE, name + ".raw")
    shard = wm[name]; h = header(shard); m = h[name]
    if os.path.exists(p):
        raw = open(p, "rb").read()
    else:
        a, b = m["data_offsets"]; off = h["__off__"]
        raw = _get(BASE + shard, off + a, off + b - 1)
        if cache: open(p, "wb").write(raw)
    if m["dtype"] == "BF16": arr = bf16_to_f32(raw)
    elif m["dtype"] == "F32": arr = np.frombuffer(raw, dtype=np.float32)
    else: raise ValueError(m["dtype"])
    return arr.reshape(m["shape"]).astype(dtype, copy=False)

def fetch_many(names, workers=16):
    wm = weight_map()
    with cf.ThreadPoolExecutor(workers) as ex:
        return dict(zip(names, ex.map(lambda n: fetch(n, wm), names)))

def fetch_rows(name, rows, workers=32):
    """Fetch selected rows of a 2-D bf16 tensor (e.g. embedding rows for the tokens in use)."""
    wm = weight_map(); shard = wm[name]; h = header(shard); m = h[name]
    ncol = m["shape"][1]; a0 = h["__off__"] + m["data_offsets"][0]; rb = 2 * ncol
    rows = list(rows)
    with cf.ThreadPoolExecutor(workers) as ex:
        raws = list(ex.map(lambda r: _get(BASE + shard, a0 + r * rb, a0 + (r + 1) * rb - 1), rows))
    return {r: bf16_to_f32(x) for r, x in zip(rows, raws)}
