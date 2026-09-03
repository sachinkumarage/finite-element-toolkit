"""Material property data models for the Finite Element Toolkit."""

from femtoolkit.materials.finite_strain import SaintVenantKirchhoff1D, SaintVenantKirchhoff3D
from femtoolkit.materials.hardening import (
    BilinearIsotropicHardeningMaterial1D,
    BilinearKinematicHardeningMaterial1D,
    DecoupledIsotropicHardeningAdapter2D,
    MultilinearIsotropicHardeningMaterial1D,
)
from femtoolkit.materials.j2_plasticity import J2Plasticity3D
from femtoolkit.materials.linear_elastic_2d import LinearElastic2D
from femtoolkit.materials.linear_elastic_3d import LinearElastic3D
from femtoolkit.materials.material import Material
from femtoolkit.materials.nonlinear import (
    ElasticMaterialAdapter,
    ElasticPerfectlyPlasticMaterial1D,
    MaterialState,
    NonlinearMaterial,
)

__all__ = [
    "BilinearIsotropicHardeningMaterial1D",
    "BilinearKinematicHardeningMaterial1D",
    "DecoupledIsotropicHardeningAdapter2D",
    "ElasticMaterialAdapter",
    "ElasticPerfectlyPlasticMaterial1D",
    "J2Plasticity3D",
    "LinearElastic2D",
    "LinearElastic3D",
    "Material",
    "MaterialState",
    "MultilinearIsotropicHardeningMaterial1D",
    "NonlinearMaterial",
    "SaintVenantKirchhoff1D",
    "SaintVenantKirchhoff3D",
]
