"""Synthetic example. Run after installing the package with `pip install .`."""
import json
import numpy as np
from ta import TA, metrics


def main():
    rng = np.random.RandomState(2026)
    truth = np.repeat(np.arange(3), [45, 60, 75])
    members = np.tile(truth[:, None], (1, 20))
    noisy = rng.rand(*members.shape) < 0.20
    members[noisy] = rng.randint(0, 3, size=noisy.sum())
    # Illustrative asymmetric geometry; these are not benchmark-selected values.
    model = TA(n_clusters=3, lambda_=1e-3, gamma=2.0,
                   delta=0.25, eta=0.5, seed=2026)
    prediction = model.fit_predict(members)
    print(json.dumps({
        "cluster_sizes": model.cluster_sizes_.tolist(),
        "partition_weights": model.alpha_.tolist(),
        "objective_history": model.objective_.tolist(),
        "metrics": metrics(prediction, truth),
    }, indent=2))


if __name__ == "__main__":
    main()
