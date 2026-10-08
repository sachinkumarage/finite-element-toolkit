"""Example: surrogate models on analytical benchmark functions (Version 35).

**Procedure.** Trains a
:class:`~femtoolkit.surrogate.models.polynomial.PolynomialRegressionSurrogate`
and a :class:`~femtoolkit.surrogate.models.rbf.RBFSurrogate` on three
mathematical functions with a known, exact solution, independent of any
FEA simulation:

- ``y = x^2`` (1D, exactly reproduced by a degree-2 polynomial)
- ``y = sin(x)`` (1D, smooth but not polynomial -- a good RBF case)
- ``y = x1^2 + x2^2`` (2D, exactly reproduced by a degree-2 polynomial)

No :class:`~femtoolkit.application.project.Project` or FEA solve is
involved -- these benchmarks exist purely to verify the surrogate
machinery itself (fitting, prediction, scaling, metrics, applicability
domain) against functions whose correct answer is known in closed form,
before it is ever trusted on real simulation data.
"""

from __future__ import annotations

import numpy as np

from femtoolkit.surrogate.datasets import Snapshot, SnapshotDataset
from femtoolkit.surrogate.models import PolynomialRegressionSurrogate, RBFSurrogate


def _dataset_from_function(
    feature_names: list[str],
    response_name: str,
    points: np.ndarray,
    function: callable,
    dataset_id: str,
) -> SnapshotDataset:
    dataset = SnapshotDataset(
        feature_names=feature_names, response_names=[response_name], dataset_id=dataset_id
    )
    for index, point in enumerate(points):
        inputs = dict(zip(feature_names, point, strict=True))
        dataset.add_snapshot(
            Snapshot(
                snapshot_id=f"{dataset_id}-{index}",
                inputs=inputs,
                outputs={response_name: function(*point)},
            )
        )
    return dataset


def run_square_benchmark() -> None:
    print("\n=== Benchmark: y = x^2 (polynomial regression) ===")
    x = np.linspace(-3.0, 3.0, 13).reshape(-1, 1)
    dataset = _dataset_from_function(["x"], "y", x, lambda v: v**2, "benchmark-square")

    model = PolynomialRegressionSurrogate(degree=2)
    x_train, y_train = dataset.to_arrays()
    model.fit(
        x_train, y_train, feature_names=["x"], response_names=["y"], dataset_id=dataset.dataset_id
    )

    probe = np.array([[0.5], [1.5], [-2.0]])
    predicted = model.predict(probe)
    expected = probe**2
    max_error = np.max(np.abs(predicted - expected))
    print(f"Probe points: {probe.ravel().tolist()}")
    print(f"Predicted:    {predicted.ravel().tolist()}")
    print(f"Expected:     {expected.ravel().tolist()}")
    print(f"Max abs error: {max_error:.3e} (expect ~0, exact for degree>=2)")


def run_sine_benchmark() -> None:
    print("\n=== Benchmark: y = sin(x) (RBF surrogate) ===")
    x = np.linspace(-np.pi, np.pi, 21).reshape(-1, 1)
    dataset = _dataset_from_function(["x"], "y", x, lambda v: np.sin(v), "benchmark-sine")

    model = RBFSurrogate(kernel="gaussian", epsilon=1.0)
    x_train, y_train = dataset.to_arrays()
    model.fit(
        x_train, y_train, feature_names=["x"], response_names=["y"], dataset_id=dataset.dataset_id
    )

    probe = np.linspace(-2.5, 2.5, 7).reshape(-1, 1)
    predicted = model.predict(probe)
    expected = np.sin(probe)
    print(f"Max abs error on a dense probe grid: {np.max(np.abs(predicted - expected)):.3e}")


def run_bivariate_benchmark() -> None:
    print("\n=== Benchmark: y = x1^2 + x2^2 (polynomial regression, 2D) ===")
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 2.0, size=(30, 2))
    dataset = _dataset_from_function(
        ["x1", "x2"], "y", x, lambda a, b: a**2 + b**2, "benchmark-bivariate-square"
    )

    model = PolynomialRegressionSurrogate(degree=2)
    x_train, y_train = dataset.to_arrays()
    model.fit(
        x_train,
        y_train,
        feature_names=["x1", "x2"],
        response_names=["y"],
        dataset_id=dataset.dataset_id,
    )

    probe = np.array([[1.0, 1.0], [-1.5, 0.5], [0.0, 0.0]])
    predicted = model.predict(probe)
    expected = (probe**2).sum(axis=1, keepdims=True)
    max_error = np.max(np.abs(predicted - expected))
    print(f"Max abs error: {max_error:.3e} (expect ~0, exact for degree>=2)")


if __name__ == "__main__":
    run_square_benchmark()
    run_sine_benchmark()
    run_bivariate_benchmark()
