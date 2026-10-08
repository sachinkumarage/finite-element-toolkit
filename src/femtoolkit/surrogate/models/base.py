"""`SurrogateModel`: the shared abstraction behind every surrogate technique (Version 35).

.. code-block:: text

    SurrogateModel
    |-- fit()
    |-- predict()
    |-- score()
    |-- validate()
    `-- serialize()

Every concrete surrogate (:mod:`femtoolkit.surrogate.models.polynomial`,
:mod:`femtoolkit.surrogate.models.rbf`) owns its own feature/response
scaling (:mod:`femtoolkit.surrogate.scaling`, fit on training data
only) and its own :class:`~femtoolkit.surrogate.domain.ApplicabilityDomain`
(built from the raw training inputs), so a prediction is always
returned in physical units together with an explicit domain status --
a caller never has to remember to invert a transform or check bounds
itself. :meth:`SurrogateModel.predict_point` is the one entry point
that makes it obvious a result is a *surrogate* prediction, not a
direct FEA result (spec section 23).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np

from femtoolkit.exceptions import ValidationError
from femtoolkit.surrogate.domain import ApplicabilityDomain, DomainStatus
from femtoolkit.surrogate.metrics import MetricSet, compute_metrics
from femtoolkit.surrogate.scaling import Scaler, build_scaler


class PredictionStatus(Enum):
    """Whether a surrogate prediction was actually computed.

    Attributes:
        OK: The prediction was computed normally (regardless of its
            applicability-domain status, which is reported separately).
        INVALID_INPUT: The query point was missing a feature, carried
            an unexpected one, or contained a non-finite value -- no
            prediction was computed.
    """

    OK = "ok"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class SurrogatePrediction:
    """One surrogate prediction, together with the metadata that qualifies it (spec section 23).

    Attributes:
        values: The predicted response value(s), in physical units,
            keyed by response name. Empty if ``status`` is
            :attr:`PredictionStatus.INVALID_INPUT`.
        model_name: The surrogate model type's name (e.g. ``"polynomial"``).
        model_version: The training metadata's model version identifier.
        dataset_id: The identifier of the dataset this model was trained on.
        dataset_version: The version of that dataset at training time.
        input_point: The query point's feature values, exactly as supplied.
        scaling_config: The feature/response scaler type names used.
        status: Whether a prediction was actually computed.
        domain_status: The query point's :class:`~femtoolkit.surrogate.domain.DomainStatus`.
        warnings: Human-readable warnings (e.g. an out-of-domain feature).
        is_surrogate_prediction: Always ``True`` -- an explicit, named
            flag so a report or log line can never mistake this for a
            direct high-fidelity FEA result.
    """

    values: dict[str, float]
    model_name: str
    model_version: str
    dataset_id: str
    dataset_version: int
    input_point: dict[str, float]
    scaling_config: dict[str, str]
    status: PredictionStatus
    domain_status: DomainStatus
    warnings: list[str] = field(default_factory=list)
    is_surrogate_prediction: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable representation of this prediction."""
        return {
            "values": dict(self.values),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "input_point": dict(self.input_point),
            "scaling_config": dict(self.scaling_config),
            "status": self.status.value,
            "domain_status": self.domain_status.value,
            "warnings": list(self.warnings),
            "is_surrogate_prediction": self.is_surrogate_prediction,
        }


@dataclass
class TrainingMetadata:
    """Reproducibility metadata recorded once, at :meth:`SurrogateModel.fit` time.

    Attributes:
        dataset_id: The training dataset's identifier.
        dataset_version: The training dataset's version at training time.
        feature_names: The input feature names, in training order.
        response_names: The output response names, in training order.
        n_training_samples: How many rows were actually used to fit the model.
        random_seed: The random seed used during training, if any.
        model_version: A model-version identifier, bumped whenever this
            model is retrained (``str(int(time))``-independent -- simply
            a monotonically increasing counter supplied by the caller,
            defaulting to ``"1"``).
        software_version: The Finite Element Toolkit version that produced this model.
        created_at: ISO-8601 UTC timestamp when the model was fit.
    """

    dataset_id: str
    dataset_version: int
    feature_names: list[str]
    response_names: list[str]
    n_training_samples: int
    random_seed: int | None = None
    model_version: str = "1"
    software_version: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class SurrogateModel(ABC):
    """The shared, transparent abstraction every surrogate technique implements.

    Attributes:
        name: A short, stable type name for this surrogate (e.g.
            ``"polynomial"``, ``"rbf"``), used in persistence and reports.
    """

    name: str = "surrogate"

    def __init__(
        self, feature_scaler_type: str = "standard", response_scaler_type: str = "standard"
    ) -> None:
        """Create an unfitted surrogate.

        Args:
            feature_scaler_type: One of
                :data:`~femtoolkit.surrogate.scaling.SCALER_TYPES`,
                applied to inputs before fitting/predicting.
            response_scaler_type: Likewise, applied to outputs.
        """
        self._feature_scaler_type = feature_scaler_type
        self._response_scaler_type = response_scaler_type
        self._feature_scaler: Scaler | None = None
        self._response_scaler: Scaler | None = None
        self._domain: ApplicabilityDomain | None = None
        self.training_metadata: TrainingMetadata | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return self.training_metadata is not None

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ValidationError(f"{type(self).__name__} must be fit() before use.")

    @abstractmethod
    def _fit_scaled(self, x_scaled: np.ndarray, y_scaled: np.ndarray) -> None:
        """Fit this model's own parameters on already-scaled ``(x, y)``."""
        raise NotImplementedError

    @abstractmethod
    def _predict_scaled(self, x_scaled: np.ndarray) -> np.ndarray:
        """Predict already-scaled responses from already-scaled inputs."""
        raise NotImplementedError

    @abstractmethod
    def model_parameters(self) -> dict[str, Any]:
        """Return this model's fitted, model-specific parameters as plain JSON-serializable data."""
        raise NotImplementedError

    @abstractmethod
    def load_model_parameters(self, parameters: dict[str, Any]) -> None:
        """Restore this model's fitted, model-specific parameters from :meth:`model_parameters`."""
        raise NotImplementedError

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        response_names: list[str],
        *,
        dataset_id: str = "",
        dataset_version: int = 1,
        random_seed: int | None = None,
        model_version: str = "1",
    ) -> SurrogateModel:
        """Fit this surrogate to training data, in physical units.

        Args:
            x: Training inputs, shape ``(n_samples, n_features)``.
            y: Training outputs, shape ``(n_samples, n_responses)``.
            feature_names: Each column of ``x``'s name, in order.
            response_names: Each column of ``y``'s name, in order.
            dataset_id: The training dataset's identifier, recorded in
                :attr:`training_metadata` and every subsequent prediction's metadata.
            dataset_version: The training dataset's version.
            random_seed: The random seed used for this training run, if any.
            model_version: A caller-supplied model version identifier.

        Returns:
            ``self``, for chaining.

        Raises:
            ValidationError: If ``x``/``y`` have mismatched row counts,
                fewer rows than :attr:`feature_names` has entries (for
                models that need it -- enforced by the concrete
                subclass), or contain non-finite values.
        """
        from femtoolkit.config import __version__ as software_version

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or y.ndim != 2:
            raise ValidationError("fit() requires 2D x and y arrays.")
        if x.shape[0] != y.shape[0]:
            raise ValidationError(
                f"x has {x.shape[0]} rows but y has {y.shape[0]}; they must match."
            )
        if x.shape[1] != len(feature_names) or y.shape[1] != len(response_names):
            raise ValidationError("x/y column counts must match feature_names/response_names.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValidationError("fit() received non-finite values in x or y.")

        self._feature_scaler = build_scaler(self._feature_scaler_type)
        self._response_scaler = build_scaler(self._response_scaler_type)
        x_scaled = self._feature_scaler.fit_transform(x)
        y_scaled = self._response_scaler.fit_transform(y)

        self._fit_scaled(x_scaled, y_scaled)
        self._domain = ApplicabilityDomain.from_training_data(x, feature_names)
        self.training_metadata = TrainingMetadata(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            feature_names=list(feature_names),
            response_names=list(response_names),
            n_training_samples=x.shape[0],
            random_seed=random_seed,
            model_version=model_version,
            software_version=software_version,
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict responses for a batch of query points, in physical units.

        Args:
            x: Query inputs, shape ``(n_queries, n_features)``, columns
                ordered exactly as :attr:`training_metadata`'s ``feature_names``.

        Returns:
            Predicted responses, shape ``(n_queries, n_responses)``, in
            physical units (already inverse-scaled) -- never the raw
            normalized model output.
        """
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        x_scaled = self._feature_scaler.transform(x)
        y_scaled = self._predict_scaled(x_scaled)
        return self._response_scaler.inverse_transform(y_scaled)

    def predict_point(self, point: dict[str, float]) -> SurrogatePrediction:
        """Predict one query point, returning a fully-labeled :class:`SurrogatePrediction`.

        This is the entry point application code should prefer over
        :meth:`predict` whenever the result may reach an engineering
        user or a report -- it always carries the applicability-domain
        status and model/dataset identity alongside the predicted
        value(s) (spec section 23/24), so a surrogate result is never
        mistaken for a direct FEA result.

        Args:
            point: The query point's feature values, keyed by name.

        Returns:
            A :class:`SurrogatePrediction`.
        """
        self._require_fitted()
        metadata = self.training_metadata
        scaling_config = {
            "feature_scaler": self._feature_scaler_type,
            "response_scaler": self._response_scaler_type,
        }
        domain_result = self._domain.check(point)
        if domain_result.status is DomainStatus.INVALID:
            return SurrogatePrediction(
                values={},
                model_name=self.name,
                model_version=metadata.model_version,
                dataset_id=metadata.dataset_id,
                dataset_version=metadata.dataset_version,
                input_point=dict(point),
                scaling_config=scaling_config,
                status=PredictionStatus.INVALID_INPUT,
                domain_status=domain_result.status,
                warnings=domain_result.warnings,
            )

        x_row = np.array([[point[name] for name in metadata.feature_names]], dtype=float)
        predicted = self.predict(x_row)[0]
        values = {name: float(predicted[i]) for i, name in enumerate(metadata.response_names)}
        return SurrogatePrediction(
            values=values,
            model_name=self.name,
            model_version=metadata.model_version,
            dataset_id=metadata.dataset_id,
            dataset_version=metadata.dataset_version,
            input_point=dict(point),
            scaling_config=scaling_config,
            status=PredictionStatus.OK,
            domain_status=domain_result.status,
            warnings=domain_result.warnings,
        )

    def score(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute each response's :math:`R^2` score against held-out data.

        Args:
            x: Query inputs, shape ``(n_samples, n_features)``.
            y: True outputs, shape ``(n_samples, n_responses)``.

        Returns:
            :math:`R^2`, keyed by response name.
        """
        self._require_fitted()
        predicted = self.predict(x)
        y = np.asarray(y, dtype=float)
        return {
            name: compute_metrics(y[:, i], predicted[:, i]).r2
            for i, name in enumerate(self.training_metadata.response_names)
        }

    def validate(self, x: np.ndarray, y: np.ndarray) -> dict[str, MetricSet]:
        """Compute the full :class:`~femtoolkit.surrogate.metrics.MetricSet` per response.

        A surrogate model must never be evaluated only on its own
        training data -- callers should pass a held-out
        validation/test split here (see
        :mod:`femtoolkit.surrogate.workflows.training`).

        Args:
            x: Query inputs, shape ``(n_samples, n_features)``.
            y: True outputs, shape ``(n_samples, n_responses)``.

        Returns:
            A :class:`~femtoolkit.surrogate.metrics.MetricSet`, keyed by response name.
        """
        self._require_fitted()
        predicted = self.predict(x)
        y = np.asarray(y, dtype=float)
        return {
            name: compute_metrics(y[:, i], predicted[:, i])
            for i, name in enumerate(self.training_metadata.response_names)
        }

    def serialize(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable dict describing this fitted model.

        Used by :mod:`femtoolkit.surrogate.persistence` -- never a
        pickle of ``self``. See that module for the full on-disk schema
        (this model's parameters plus the scalers, applicability
        domain, and training metadata).
        """
        self._require_fitted()
        metadata = self.training_metadata
        return {
            "model_name": self.name,
            "feature_scaler_type": self._feature_scaler_type,
            "response_scaler_type": self._response_scaler_type,
            "feature_scaler": self._feature_scaler.to_dict(),
            "response_scaler": self._response_scaler.to_dict(),
            "domain": self._domain.to_dict(),
            "training_metadata": {
                "dataset_id": metadata.dataset_id,
                "dataset_version": metadata.dataset_version,
                "feature_names": list(metadata.feature_names),
                "response_names": list(metadata.response_names),
                "n_training_samples": metadata.n_training_samples,
                "random_seed": metadata.random_seed,
                "model_version": metadata.model_version,
                "software_version": metadata.software_version,
                "created_at": metadata.created_at,
            },
            "model_parameters": self.model_parameters(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore this model's full state from :meth:`serialize`'s output.

        Only called by :mod:`femtoolkit.surrogate.persistence` after it
        has already validated the on-disk schema/software version --
        this method itself assumes ``state`` is well-formed.
        """
        self._feature_scaler_type = state["feature_scaler_type"]
        self._response_scaler_type = state["response_scaler_type"]
        self._feature_scaler = Scaler.from_dict(state["feature_scaler"])
        self._response_scaler = Scaler.from_dict(state["response_scaler"])
        self._domain = ApplicabilityDomain.from_dict(state["domain"])
        metadata = state["training_metadata"]
        self.training_metadata = TrainingMetadata(
            dataset_id=metadata["dataset_id"],
            dataset_version=metadata["dataset_version"],
            feature_names=list(metadata["feature_names"]),
            response_names=list(metadata["response_names"]),
            n_training_samples=metadata["n_training_samples"],
            random_seed=metadata.get("random_seed"),
            model_version=metadata.get("model_version", "1"),
            software_version=metadata.get("software_version", ""),
            created_at=metadata.get("created_at", ""),
        )
        self.load_model_parameters(state["model_parameters"])


__all__ = [
    "PredictionStatus",
    "SurrogateModel",
    "SurrogatePrediction",
    "TrainingMetadata",
]
