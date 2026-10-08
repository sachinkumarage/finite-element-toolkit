"""Training a surrogate for the model discrepancy ``delta(x) = y_H(x) - y_L(x)`` (Version 37).

.. math::

    \\delta(x) = y_H(x) - y_L(x)

The discrepancy is itself just another scalar-valued function of the design
variables -- so it is trained with the *unmodified* Version 35 surrogate
machinery (:func:`~femtoolkit.surrogate.workflows.training.train_surrogate`,
:data:`~femtoolkit.surrogate.models.SURROGATE_MODEL_TYPES`), on the dataset
:meth:`~femtoolkit.multifidelity.dataset.MultiFidelityDataset.to_discrepancy_dataset`
builds. No new regression or machine-learning code is introduced here.
"""

from __future__ import annotations

from femtoolkit.multifidelity.dataset import MultiFidelityDataset
from femtoolkit.surrogate.models.base import SurrogateModel
from femtoolkit.surrogate.validation import SurrogateValidationReport
from femtoolkit.surrogate.workflows.training import TrainingConfig, train_surrogate


def train_discrepancy_surrogate(
    dataset: MultiFidelityDataset, config: TrainingConfig | None = None
) -> tuple[SurrogateModel, SurrogateValidationReport]:
    """Train a surrogate model of the discrepancy ``x -> delta(x)`` on paired samples.

    Args:
        dataset: The paired low-/high-fidelity dataset. At least one sample must be
            paired (see
            :meth:`~femtoolkit.multifidelity.dataset.MultiFidelityDataset.paired_samples`).
        config: The surrogate training configuration (model type, scaling, split
            fractions, ...). ``None`` uses every
            :class:`~femtoolkit.surrogate.workflows.training.TrainingConfig` default
            (a polynomial surrogate).

    Returns:
        A ``(discrepancy_model, report)`` pair: the fitted discrepancy surrogate
        and its Version 35 validation report (held-out metrics, never training-only).

    Raises:
        ValidationError: If no sample in ``dataset`` is paired yet.
        InsufficientSnapshotsError: If too few paired samples exist to fit the
            configured surrogate.
    """
    discrepancy_dataset = dataset.to_discrepancy_dataset()
    return train_surrogate(discrepancy_dataset, config)


__all__ = ["train_discrepancy_surrogate"]
