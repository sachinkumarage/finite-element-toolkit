"""Material property data models for the Finite Element Toolkit."""

from femtoolkit.materials.finite_strain import SaintVenantKirchhoff1D, SaintVenantKirchhoff3D
from femtoolkit.materials.finite_strain_plasticity import (
    FiniteStrainPlasticMaterial,
    J2FiniteStrainPlasticity3D,
)
from femtoolkit.materials.hardening import (
    BilinearIsotropicHardeningMaterial1D,
    BilinearKinematicHardeningMaterial1D,
    DecoupledIsotropicHardeningAdapter2D,
    MultilinearIsotropicHardeningMaterial1D,
)
from femtoolkit.materials.hyperelastic import HyperelasticMaterial
from femtoolkit.materials.j2_plasticity import J2Plasticity3D
from femtoolkit.materials.linear_elastic_2d import LinearElastic2D
from femtoolkit.materials.linear_elastic_3d import LinearElastic3D
from femtoolkit.materials.material import Material
from femtoolkit.materials.mooney_rivlin import MooneyRivlin3D
from femtoolkit.materials.neo_hookean import NeoHookean3D
from femtoolkit.materials.nonlinear import (
    ElasticMaterialAdapter,
    ElasticPerfectlyPlasticMaterial1D,
    MaterialState,
    NonlinearMaterial,
)
from femtoolkit.materials.thermal_properties import (
    TemperatureDependentProperty,
    ThermalProperty,
    evaluate_thermal_property,
)
from femtoolkit.materials.thermoelastic import (
    ThermoelasticMaterial3D,
    ThermoelasticMaterialAtTemperature,
)

__all__ = [
    "BilinearIsotropicHardeningMaterial1D",
    "BilinearKinematicHardeningMaterial1D",
    "DecoupledIsotropicHardeningAdapter2D",
    "ElasticMaterialAdapter",
    "ElasticPerfectlyPlasticMaterial1D",
    "FiniteStrainPlasticMaterial",
    "HyperelasticMaterial",
    "J2FiniteStrainPlasticity3D",
    "J2Plasticity3D",
    "LinearElastic2D",
    "LinearElastic3D",
    "Material",
    "MaterialState",
    "MooneyRivlin3D",
    "MultilinearIsotropicHardeningMaterial1D",
    "NeoHookean3D",
    "NonlinearMaterial",
    "SaintVenantKirchhoff1D",
    "SaintVenantKirchhoff3D",
    "TemperatureDependentProperty",
    "ThermalProperty",
    "ThermoelasticMaterial3D",
    "ThermoelasticMaterialAtTemperature",
    "evaluate_thermal_property",
]
