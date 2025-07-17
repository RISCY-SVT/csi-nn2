#!/usr/bin/env python3
"""
check_conv2d_dat.py – verify tensors packed in tests/unit_test/valid_data/conv2d.dat

• parses C-style unsigned-char arrays
• reconstructs float32/float16 tensors
• validates size against // comments
• for _in/_ker/_bias pairs recomputes convolution (pad=1, stride=1)
  and compares with *_out  (KL + cosine + per-element diff)
"""

import re, sys, math, struct, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy

# ---------- helpers -----------------------------------------------------------
HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
DEC_RE = re.compile(r"\b\d+\b")
ARR_RE = re.compile(r"unsigned char\s+(\w+)\s*\[\]\s*=\s*\{([^}]*)\}", re.S)
DIM_RE = re.compile(r"//\s*(\w+)\s*:\s*\[([^\]]+)\]")

def bytes_from_brace(blob: str):
    nums = HEX_RE.findall(blob) + [
        n for n in DEC_RE.findall(blob) if n not in HEX_RE.findall(blob)
    ]
    return bytes(int(x, 16) if x.startswith("0x") else int(x) for x in nums)

def float_from_bytes(b: bytes, dtype: str):
    dt = np.float32 if dtype == "fp32" else np.float16
    return np.frombuffer(b, dtype=dt)

def conv2d_nchw(x, w, bias, pad=1, stride=1, dil=1):
    n, ic, ih, iw = x.shape
    oc, _, kh, kw = w.shape
    oh = (ih + 2*pad - dil*(kh-1) - 1)//stride + 1
    ow = (iw + 2*pad - dil*(kw-1) - 1)//stride + 1
    y = np.zeros((n, oc, oh, ow), x.dtype)
    xpad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
    for o in range(oc):
        for i in range(ic):
            for r in range(kh):
                for c in range(kw):
                    y[:,o] += w[o,i,r,c] * \
                        xpad[:,i,
                             r*dil : r*dil+oh*stride : stride,
                             c*dil : c*dil+ow*stride : stride]
        y[:,o] += bias[o]
    return y

def metrics(ref, out):
    r = ref.flatten().astype(np.float32)
    o = out.flatten().astype(np.float32)
    return entropy(r+1e-12, o+1e-12), 1 - cosine(r, o)

# ---------- parse whole file --------------------------------------------------
FILE = Path("tests/unit_test/valid_data/conv2d.dat")
text = FILE.read_text()

# gather expected dims from comments
dims_map = {}
for lbl, vec in DIM_RE.findall(text):
    dims_map[lbl.lower()] = tuple(int(x) for x in vec.split(','))

# parse arrays
data = {}
for name, blob in ARR_RE.findall(text):
    raw = bytes_from_brace(blob)
    data[name] = raw

# ---------- scenarios ---------------------------------------------------------
SCENARIOS = [
    "conv2d1x1s1",
    "conv2d_im2col",
    "conv2d_winograd",
]

def check_one(scn, fp):
    prefix = f"{scn}_{fp}"
    dtype = "fp32" if fp=="fp32" else "fp16"
    print(f"\n=== [{prefix}] ===")
    needed = ["in","ker","ker1","bias","out"]
    missing = [k for k in needed if f"{prefix}_{k}" not in data]
    if missing:
        print(" missing arrays:", ", ".join(missing))
        return False

    tensors = {}
    for k in needed:
        arr = data[f"{prefix}_{k}"]
        shape = dims_map.get(k)  # uses 'input','kernel',etc.
        if k == "in":   key = "input"
        elif k.startswith("ker"): key = "kernel"
        elif k=="bias": key = "bias"
        else:           key = "output"
        shape = dims_map.get(key, ())
        if not shape:
            print(f"  [WARN] dims for {k} not found in comments")
        t = float_from_bytes(arr, dtype)
        if shape and t.size != math.prod(shape):
            print(f"  size mismatch {k}: have {t.size}, expect {shape}")
        tensors[k] = t.reshape(shape) if shape else t

    # recompute conv
    ref = conv2d_nchw(tensors["in"].astype(np.float32),
                      tensors["ker"].astype(np.float32),
                      tensors["bias"].astype(np.float32))

    kl, cs = metrics(tensors["out"], ref)
    print(f" KL={kl:.6f}  cos={cs:.6f}")

    diff = np.abs(tensors["out"].astype(np.float32)-ref)
    bad  = np.argwhere(diff>1e-4)
    if bad.size:
        print(f"  mismatches: {bad.shape[0]}  (showing first 6)")
        for idx in bad[:6]:
            r=tuple(idx); print("   ",r,
                tensors["out"][r], ref[r], diff[r])
        return False
    print("  OK")
    return True

overall = True
for scn in SCENARIOS:
    for fp in ("fp32","fp16"):
        overall &= check_one(scn, fp)

print("\nSUMMARY:", "PASS" if overall else "FAIL")
sys.exit(0 if overall else 1)
