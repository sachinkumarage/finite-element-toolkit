"""Parameter sweeps: auto-generating scenarios from parameter value lists (Version 30).

**Engineering concept.** A *parameter study* asks "how does the result
change as this parameter (or these parameters) vary?" rather than
requiring an engineer to hand-write one :class:`~femtoolkit.studies.scenarios.Scenario`
per value. :func:`generate_scenarios` takes one or more
:class:`ParameterDefinition` value lists and produces one scenario per
combination -- the Cartesian product -- so a single-parameter sweep
(one definition) and a multi-parameter study (several) are the exact
same code path.

**Combinatorial growth warning.** The scenario count is the *product* of
every parameter's value count: 3 parameters with 5 values each is
already 125 scenarios, not 15. This grows fast, and each scenario is one
full FEA solve. :func:`generate_scenarios` therefore enforces
``max_scenarios`` *before* generating anything -- see
:data:`DEFAULT_MAX_SCENARIOS` and :class:`~femtoolkit.exceptions.exceptions.StudySizeExceededError`.
A study is never silently truncated; if the limit is hit, reduce the
number of parameters, the number of values per parameter, or raise
``max_scenarios`` deliberately.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from femtoolkit.exceptions import StudySizeExceededError, ValidationError
from femtoolkit.studies.scenarios import Scenario

DEFAULT_MAX_SCENARIOS = 100
"""The default ceiling on generated scenarios, chosen so a parameter
sweep never silently launches an unreasonable number of FEA solves."""


@dataclass
class ParameterDefinition:
    """One parameter to vary in a parameter study.

    Attributes:
        path: The dotted override path this parameter controls (see
            :func:`~femtoolkit.studies.scenarios.apply_scenario`), e.g.
            ``"loads.0.magnitude"``.
        label: A short, human-readable name for this parameter (e.g.
            ``"Tip load (N)"``), used in scenario names and study
            reports/plots.
        values: The values to sweep this parameter over. Must be
            non-empty.
    """

    path: str
    label: str
    values: list[Any]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValidationError(f"Parameter {self.label!r} ({self.path!r}) has no values.")


def count_combinations(parameters: list[ParameterDefinition]) -> int:
    """Return how many scenarios :func:`generate_scenarios` would produce.

    This is the product of every parameter's value count -- callers can
    use it to check a study's size *before* calling
    :func:`generate_scenarios`.

    Args:
        parameters: The parameter definitions to combine.

    Returns:
        The Cartesian-product scenario count (``1`` if ``parameters`` is
        empty).
    """
    total = 1
    for parameter in parameters:
        total *= len(parameter.values)
    return total


def generate_scenarios(
    base_name: str,
    parameters: list[ParameterDefinition],
    max_scenarios: int = DEFAULT_MAX_SCENARIOS,
) -> list[Scenario]:
    """Generate one scenario per combination of every parameter's values.

    Args:
        base_name: A short prefix used to build each scenario's name and
            ID (e.g. ``"Load Study"`` -> ``"Load Study #0"``).
        parameters: The parameters to sweep. A single entry produces a
            simple single-parameter sweep; several entries produce the
            full Cartesian product (a multi-parameter study).
        max_scenarios: The maximum number of scenarios this call may
            produce. Checked *before* any scenario is built.

    Returns:
        One :class:`~femtoolkit.studies.scenarios.Scenario` per
        combination, in Cartesian-product order (the last parameter
        varies fastest), with unique, deterministic ``scenario_id``s.

    Raises:
        ValidationError: If ``parameters`` is empty.
        StudySizeExceededError: If the combination count exceeds
            ``max_scenarios``. The message explains the actual count and
            how to reduce it -- the study is never silently truncated.
    """
    if not parameters:
        raise ValidationError("generate_scenarios requires at least one ParameterDefinition.")

    total = count_combinations(parameters)
    if total > max_scenarios:
        breakdown = ", ".join(f"{p.label!r} ({len(p.values)} values)" for p in parameters)
        raise StudySizeExceededError(
            f"Parameter sweep would generate {total} scenarios ({breakdown}), which exceeds "
            f"max_scenarios={max_scenarios}. Reduce the number of parameters, the number of "
            "values per parameter, or explicitly raise max_scenarios if this many scenarios "
            "is intentional."
        )

    value_lists = [parameter.values for parameter in parameters]
    scenarios: list[Scenario] = []
    for index, combination in enumerate(itertools.product(*value_lists)):
        overrides = {
            parameter.path: value for parameter, value in zip(parameters, combination, strict=True)
        }
        description = ", ".join(
            f"{parameter.label}={value!r}" for parameter, value in zip(
                parameters, combination, strict=True
            )
        )
        scenarios.append(
            Scenario(
                scenario_id=f"{base_name}-{index}",
                name=f"{base_name} #{index}",
                description=description,
                parameter_overrides=overrides,
            )
        )
    return scenarios
