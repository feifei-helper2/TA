
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def relabel0(labels):
    _, inv = np.unique(np.asarray(labels).ravel(), return_inverse=True)
    return inv.astype(np.int32)

def contingency(pred, gt):
    pred, gt = relabel0(pred), relabel0(gt)
    C = np.zeros((pred.max()+1, gt.max()+1), dtype=np.int64)
    np.add.at(C, (pred, gt), 1)
    return C

def acc_score(pred, gt):
    C = contingency(pred, gt)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].sum() / C.sum())

def purity_score(pred, gt):
    C = contingency(pred, gt)
    return float(C.max(axis=1).sum() / C.sum())

def pair_prf(pred, gt):
    C = contingency(pred, gt)
    comb2 = lambda a: a * (a - 1) / 2.0
    inter = float(np.sum(comb2(C)))
    pden = float(np.sum(comb2(C.sum(axis=1))))
    rden = float(np.sum(comb2(C.sum(axis=0))))
    precision = inter / pden if pden > 0 else 1.0
    recall = inter / rden if rden > 0 else 1.0
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f

def metrics(pred, gt):
    p, r, f = pair_prf(pred, gt)
    return {
        "ACC": acc_score(pred, gt),
        "ARI": float(adjusted_rand_score(gt, pred)),
        "NMI": float(normalized_mutual_info_score(gt, pred, average_method="arithmetic")),
        "F": float(f),
        "Purity": purity_score(pred, gt),
        "Precision": float(p),
        "Recall": float(r),
    }
