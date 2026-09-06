"""Finite-strain thermal deformation foundation (Version 19).

Versions 17-18 already support genuinely large elastic
(:mod:`femtoolkit.materials.hyperelastic`) and plastic
(:mod:`femtoolkit.materials.finite_strain_plasticity`) deformation via
the multiplicative decomposition ``F = Fe @ Fp``. Thermal expansion is a
**third** kind of deformation -- neither elastic (stress-producing) nor
plastic (irrecoverable/dissipative) -- that a body undergoes purely
because its temperature changed. At finite strain, the natural extension
is to add a third multiplicative factor:

.. code-block:: text

    F = Fe @ Fth @ Fp

read right-to-left, exactly like ``F = Fe @ Fp`` already is: a material
point first deforms plastically (reference configuration -> a stress-free
*plastic* intermediate configuration), then thermally (that configuration
-> a further stress-free *thermal* intermediate configuration, purely a
uniform volume change), then elastically (the thermal intermediate
configuration -> the actual current one, which is what produces stress).
This specific ordering is one standard, reasonable convention (other
orderings appear in the literature); what matters for this version is
that the *elastic* part, wherever it sits in the product, is the only
factor a stress-response function should ever see.

**Isotropic thermal deformation gradient.** For isotropic thermal
expansion, `Fth` is a pure, uniform volumetric stretch -- no shape
change, no rotation, exactly mirroring
:func:`~femtoolkit.materials.thermoelastic.ThermoelasticMaterial3D.thermal_strain_voigt`'s
small-strain thermal eigenstrain, but expressed as a deformation gradient
rather than a strain:

.. code-block:: text

    Fth = lambda_th * I,    lambda_th = 1 + alpha * dT

``lambda_th`` is the **thermal stretch**: the linear (1D) thermal
expansion formula ``L/L0 = 1 + alpha*dT`` applied uniformly along every
axis (this is the standard, first-order isotropic finite-strain
generalization of the small-strain ``epsilon_thermal = alpha*dT``, and
reduces to it exactly: ``Fth = I => epsilon_thermal = 0`` at ``dT = 0``,
and for small ``alpha*dT``, ``Fth - I ~= alpha*dT*I``, matching the
small-strain thermal eigenstrain to first order).

**Scope: foundation only.** This module provides the deformation
gradient itself, and a small utility to recover the elastic part from a
known total/thermal/plastic split -- *not* a complete finite-strain
thermoplasticity return-mapping algorithm (a genuinely large
undertaking: `Fth` would need to be threaded through
:class:`~femtoolkit.materials.finite_strain_plasticity.FiniteStrainPlasticMaterial`'s
entire elastic-predictor/plastic-corrector machinery, itself an
already-involved eigendecomposition-based algorithm). What Version 19
demonstrates instead is the extension *point*: given a total deformation
gradient and known thermal (and optionally plastic) parts, the resulting
elastic deformation gradient is directly usable by any existing Version
17 hyperelastic material (:mod:`femtoolkit.materials.hyperelastic`)
completely unchanged, since those materials already accept an arbitrary
``F`` and only need it to be the *elastic* one.
"""

from __future__ import annotations

import math

import numpy as np

from femtoolkit.exceptions import InvalidDeformationGradientError

MIN_THERMAL_STRETCH: float = 1e-6
"""Minimum acceptable thermal stretch ``lambda_th``. Guards against a
physically nonsensical (non-positive, zero-or-negative-volume) thermal
contraction from an extreme cooling combined with a large positive
``alpha`` -- the finite-strain analogue of
:data:`~femtoolkit.continuum.deformation.MIN_DEFORMATION_GRADIENT_DETERMINANT`."""


def thermal_deformation_gradient(
    thermal_expansion_coefficient: float, delta_temperature: float
) -> np.ndarray:
    """Return the isotropic thermal deformation gradient ``Fth = (1 + alpha*dT) * I``.

    Args:
        thermal_expansion_coefficient: The coefficient of linear thermal
            expansion ``alpha``, in 1/K.
        delta_temperature: The temperature change ``dT = T - T_ref``, in
            kelvin.

    Returns:
        The 3x3 thermal deformation gradient.

    Raises:
        InvalidDeformationGradientError: If the resulting thermal
            stretch is not finite and positive (an extreme, physically
            invalid cooling).
    """
    thermal_stretch = 1.0 + thermal_expansion_coefficient * delta_temperature
    if not math.isfinite(thermal_stretch) or thermal_stretch < MIN_THERMAL_STRETCH:
        raise InvalidDeformationGradientError(
            f"Thermal stretch {thermal_stretch} is not physically valid (must be "
            f"finite and at least {MIN_THERMAL_STRETCH}); check alpha and dT."
        )
    return thermal_stretch * np.eye(3)


def elastic_deformation_gradient_from_thermal_split(
    total_deformation_gradient: np.ndarray,
    thermal_deformation_gradient_tensor: np.ndarray,
    plastic_deformation_gradient: np.ndarray | None = None,
) -> np.ndarray:
    """Recover ``Fe`` from ``F = Fe @ Fth @ Fp``, given the total, thermal, and plastic parts.

    ``Fe = F @ Fp^-1 @ Fth^-1`` (``Fp`` defaults to the identity -- purely
    thermoelastic deformation, no plasticity). The result is directly
    usable by any Version 17 hyperelastic material
    (:mod:`femtoolkit.materials.hyperelastic`), demonstrating the
    intended extension point (see the module docstring).

    Args:
        total_deformation_gradient: The total deformation gradient ``F``.
        thermal_deformation_gradient_tensor: The thermal deformation
            gradient ``Fth`` (see :func:`thermal_deformation_gradient`).
        plastic_deformation_gradient: The plastic deformation gradient
            ``Fp``, or ``None`` (default) for the identity (no plastic
            deformation).

    Returns:
        The 3x3 elastic deformation gradient ``Fe``.
    """
    fp = np.eye(3) if plastic_deformation_gradient is None else plastic_deformation_gradient
    return total_deformation_gradient @ np.linalg.inv(fp) @ np.linalg.inv(
        thermal_deformation_gradient_tensor
    )
