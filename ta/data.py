
from pathlib import Path
import numpy as np

DEFAULT_ORDER = [
    "MaternalHealth","CNAE9","Phishing","PSDAS","TUANDROMD","GenderGap",
    "COIL100","CrowdsourcedMapping","HTRU2","SteelEnergy",
    "Iris","Wine","BreastCancer","Digits","Penguins","MPGClass",
    "Titanic","Spector","ANES96","ModeChoice"
]

def load_benchmark(path):
    z = np.load(path, allow_pickle=False)
    out = {}
    for name in DEFAULT_ORDER:
        gk = f"gt__{name}"
        mk = f"members__{name}"
        if gk in z.files and mk in z.files:
            out[name] = {
                "gt": z[gk].astype(np.int32),
                "members": z[mk].astype(np.int32),
            }
    return out

def make_repeat_indices(pool_size=100, ensemble_size=20, repeats=10, seed=2026):
    rng = np.random.RandomState(seed)
    return np.vstack([
        rng.permutation(pool_size)[:ensemble_size]
        for _ in range(repeats)
    ]).astype(np.int32)
