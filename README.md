# TA: Tolerance-Aware Size Intervention for Discrete Ensemble Consensus

Python implementation of **TA: Tolerance-Aware Size Intervention for Discrete
Ensemble Consensus**, by YuFei Cheng, Peng Wu, and Liang Du.

TA combines normalized partition evidence with a cluster-size penalty that is
inactive within a tolerance interval. Its parameters control intervention
strength, tolerance width, and the relative weight of lower-side violations.
The same implementation supports Quad-SB, Sym-TA, CapQ, and general TA.

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/feifei-helper2/TA.git
cd TA
python -m pip install .
```

Dependencies are NumPy, SciPy, scikit-learn, and Numba. The first model run includes
Numba compilation time.

## Quick start

Run the self-contained examples:

```bash
python examples/demo.py
python examples/compare_variants.py
```

To use an ensemble of base partitions:

```python
import numpy as np
from ta import TA

members = np.load("members.npy", allow_pickle=False)
model = TA(n_clusters=3, lambda_=1e-3, gamma=2.0,
           delta=0.25, eta=0.5, seed=2026)
labels = model.fit_predict(members)
```

`members` has shape `(n_samples, n_partitions)`. Each column contains integer
cluster labels for the same samples in the same order. Labels are encoded
independently within each partition. Supply the desired number of consensus
clusters through `n_clusters`.

## Size intervention

For cluster size s and a = n/K, the penalty is

$$
\phi_{\delta,\eta}(s)
= \eta\bigl[(1-\delta)a-s\bigr]_+^2
+ \bigl[s-(1+\delta)a\bigr]_+^2.
$$

The complete size term is lambda times the sum of these penalties over clusters.

| Variant | delta | eta | Factory |
| --- | --- | --- | --- |
| Quad-SB | 0 | 1 | `make_quad_sb` |
| Sym-TA | greater than 0 | 1 | `make_sym_ta` |
| CapQ | greater than 0 | 0 | `make_capq` |
| General TA | [0, 1) | nonnegative | `make_ta` |

All factories are available from `ta`. These are nested cases: Quad-SB requires
both delta=0 and eta=1. The default delta=0.25, eta=1 selects Sym-TA; use explicit
parameters for other geometries.

| Parameter | Meaning | Domain |
| --- | --- | --- |
| `n_clusters` | Number of consensus clusters | 1 to n_samples |
| `lambda_` | Size-intervention strength | nonnegative |
| `gamma` | Exponent for adaptive partition weights | greater than 1 |
| `delta` | Relative tolerance width | [0, 1) |
| `eta` | Lower-to-upper violation weight | nonnegative |
| `seed` | Initialization seed | integer in the NumPy RandomState range |
| `max_outer`, `max_inner` | Outer iterations and coordinate sweeps | positive integers |
| `tol` | Objective stopping tolerance | positive |

The example settings illustrate the API and are not per-dataset benchmark settings.

## Outputs and evaluation

After fitting, the model exposes:

- `labels_`: consensus labels, numbered from zero.
- `cluster_sizes_`: number of samples in each consensus cluster.
- `alpha_`: learned weights for the base partitions.
- `objective_`: objective values after outer iterations.
- `balance_penalty_`: size penalty before multiplication by `lambda_`.

For labeled evaluation data:

```python
from ta import metrics
scores = metrics(labels, ground_truth)
```

The helper returns ACC, ARI, arithmetic NMI, and pairwise F, together with purity,
precision, and recall. Scores are fractions rather than percentages.

## Repository scope

This repository provides the TA solver and synthetic usage examples. Benchmark
data and dataset-specific experimental configurations are not included.
