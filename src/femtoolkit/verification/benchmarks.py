"""Analytical benchmark problems for FEA verification (Version 29).

Each function here builds a small, well-known engineering problem with
a **closed-form** analytical solution, solves it with the toolkit's
existing, unmodified analysis workflow
(:class:`~femtoolkit.analysis.static_linear.StaticLinearAnalysis`/
:class:`~femtoolkit.thermal.thermal_analysis.SteadyStateThermalAnalysis`),
and packages the comparison as one or more
:class:`~femtoolkit.verification.cases.VerificationCase` instances. No
benchmark result is ever hard-coded: every reference value is computed
from the same closed-form formula the module docstring states, using
the case's own input parameters, so changing a parameter (load,
length, modulus, ...) automatically produces the matching reference
value.

This module introduces no new element, material, or solver -- it only
*exercises* the existing ones (:class:`~femtoolkit.mesh.bar_element.BarElement`,
:class:`~femtoolkit.mesh.truss_element.TrussElement2D`,
:class:`~femtoolkit.mesh.frame_element.FrameElement2D`,
:class:`~femtoolkit.mesh.quad_element.QuadElement2D`,
:class:`~femtoolkit.mesh.cst_element.CSTElement2D`,
:class:`~femtoolkit.mesh.hex8_element.Hex8Element3D`) exactly as every
prior version's own test suite already does.
"""

from __future__ import annotations

import math

import numpy as np

from femtoolkit.analysis import (
    BoundaryCondition,
    NodalLoad,
    RotationDOF,
    StaticLinearAnalysis,
    TranslationDOF,
)
from femtoolkit.materials import LinearElastic2D, LinearElastic3D, Material
from femtoolkit.mesh import (
    BarElement,
    CSTElement2D,
    FrameElement2D,
    Hex8Element3D,
    Mesh,
    Node,
    QuadElement2D,
    TrussElement2D,
)
from femtoolkit.sections import CrossSection
from femtoolkit.thermal import PrescribedTemperature, SteadyStateThermalAnalysis, ThermalMaterial
from femtoolkit.verification.cases import VerificationCase
from femtoolkit.verification.tolerance import Tolerance

X = TranslationDOF.X
Y = TranslationDOF.Y
RZ = RotationDOF.RZ

_TIGHT_TOLERANCE = Tolerance(absolute=1e-9, relative=1e-6)
"""The default tolerance for these benchmarks: every one of them uses an
element formulation that is *exact* for the field it represents (a bar
element for uniform axial strain, a frame element for the
Euler-Bernoulli beam equation, a 1D thermal conduction element for a
linear temperature profile, Q4/CST/HEX8 for a linear displacement
field), so the FEA result should match the analytical solution to
floating-point precision, not merely "converge toward" it -- a loose
tolerance here would hide a real regression."""


def axial_bar_cases(
    load: float = 1000.0,
    area: float = 0.01,
    youngs_modulus: float = 200e9,
    length: float = 2.0,
) -> list[VerificationCase]:
    """The classical fixed-free axial bar benchmark.

    .. code-block:: text

        Fixed |---------------| Free, F
        Node 1      L          Node 2

    Analytical solution:

    .. code-block:: text

        delta = F * L / (A * E)   (tip displacement)
        sigma = F / A             (axial stress)

    Args:
        load: Axial end load ``F``, in newtons.
        area: Cross-sectional area ``A``, in square meters.
        youngs_modulus: Young's modulus ``E``, in pascals.
        length: Bar length ``L``, in meters.

    Returns:
        Two cases: tip displacement and axial stress.
    """
    analytical_displacement = load * length / (area * youngs_modulus)
    analytical_stress = load / area

    def _solve():
        material = Material(
            name="Steel", density=7850.0, youngs_modulus=youngs_modulus, poissons_ratio=0.3
        )
        section = CrossSection(area=area)
        node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        node_2 = Node(id=2, x=length, y=0.0, z=0.0)
        mesh = Mesh()
        mesh.add_node(node_1)
        mesh.add_node(node_2)
        mesh.add_element(
            BarElement(id=1, nodes=(node_1, node_2), material=material, cross_section=section)
        )

        analysis = StaticLinearAnalysis(mesh)
        analysis.add_boundary_condition(BoundaryCondition(node_1.id, dof=0, value=0.0))
        analysis.add_load(NodalLoad(node_2.id, dof=0, value=load))
        return analysis.solve()

    displacement_case = VerificationCase(
        name="Axial bar: tip displacement",
        description=(
            "Fixed-free axial bar under an end load F; closed-form displacement delta = F*L/(A*E)."
        ),
        analysis_type="linear_static",
        quantity="Tip displacement",
        reference_value=analytical_displacement,
        tolerance=_TIGHT_TOLERANCE,
        run=lambda: _solve().displacement(2, dof=0),
        units="m",
        metadata={"load": load, "area": area, "youngs_modulus": youngs_modulus, "length": length},
    )
    stress_case = VerificationCase(
        name="Axial bar: axial stress",
        description="Fixed-free axial bar under an end load F; closed-form stress sigma = F/A.",
        analysis_type="linear_static",
        quantity="Axial stress",
        reference_value=analytical_stress,
        tolerance=_TIGHT_TOLERANCE,
        run=lambda: _solve().element_stress(1),
        units="Pa",
        metadata={"load": load, "area": area, "youngs_modulus": youngs_modulus, "length": length},
    )
    return [displacement_case, stress_case]


def truss_cases(
    load: float = 1000.0,
    area: float = 0.001,
    youngs_modulus: float = 200e9,
    half_span: float = 1.0,
    height: float = 1.0,
) -> list[VerificationCase]:
    """The classical two-bar (symmetric "A-frame") truss benchmark.

    .. code-block:: text

              Apex (0, height), load F downward
               /\\
              /  \\
             /    \\
        (-half_span,0) ---- (half_span,0)
          pinned              pinned

    Statically determinate by joint equilibrium at the free apex alone
    (two members, two equilibrium equations), independent of how the two
    base supports individually share the reactions. With member length
    ``L = sqrt(half_span^2 + height^2)``:

    .. code-block:: text

        N     = -F * L / (2 * height)              (member axial force; negative = compression)
        delta = F * L^3 / (2 * height^2 * A * E)    (vertical apex displacement magnitude)

    Args:
        load: Downward apex load ``F``, in newtons.
        area: Member cross-sectional area, in square meters.
        youngs_modulus: Young's modulus, in pascals.
        half_span: Horizontal distance from the apex to each base
            support, in meters.
        height: Vertical distance from the base to the apex, in meters.

    Returns:
        Two cases: apex vertical displacement and member axial force.
    """
    member_length = math.hypot(half_span, height)
    analytical_member_force = -load * member_length / (2.0 * height)
    analytical_tip_displacement = (
        -load * member_length**3 / (2.0 * height**2 * area * youngs_modulus)
    )

    def _solve():
        material = Material(
            name="Steel", density=7850.0, youngs_modulus=youngs_modulus, poissons_ratio=0.3
        )
        section = CrossSection(area=area)
        left = Node(id=1, x=-half_span, y=0.0, z=0.0)
        right = Node(id=2, x=half_span, y=0.0, z=0.0)
        apex = Node(id=3, x=0.0, y=height, z=0.0)

        mesh = Mesh()
        for node in (left, right, apex):
            mesh.add_node(node)
        mesh.add_element(
            TrussElement2D(id=1, nodes=(left, apex), material=material, cross_section=section)
        )
        mesh.add_element(
            TrussElement2D(id=2, nodes=(right, apex), material=material, cross_section=section)
        )

        analysis = StaticLinearAnalysis(mesh)
        for node_id in (1, 2):
            analysis.add_boundary_condition(BoundaryCondition(node_id, X, 0.0))
            analysis.add_boundary_condition(BoundaryCondition(node_id, Y, 0.0))
        analysis.add_load(NodalLoad(apex.id, Y, -load))
        return analysis.solve()

    displacement_case = VerificationCase(
        name="Two-bar truss: apex vertical displacement",
        description="Symmetric two-bar truss under a vertical apex load, by joint equilibrium.",
        analysis_type="linear_static",
        quantity="Apex vertical displacement",
        reference_value=analytical_tip_displacement,
        tolerance=_TIGHT_TOLERANCE,
        run=lambda: _solve().displacement(3, dof=Y),
        units="m",
        metadata={"load": load, "area": area, "youngs_modulus": youngs_modulus},
    )
    member_force_case = VerificationCase(
        name="Two-bar truss: member axial force",
        description="Symmetric two-bar truss member axial force (negative = compression).",
        analysis_type="linear_static",
        quantity="Member axial force",
        reference_value=analytical_member_force,
        tolerance=_TIGHT_TOLERANCE,
        run=lambda: _solve().element_axial_force(1),
        units="N",
        metadata={"load": load, "area": area, "youngs_modulus": youngs_modulus},
    )
    return [displacement_case, member_force_case]


def cantilever_beam_cases(
    load: float = 1000.0,
    youngs_modulus: float = 200e9,
    moment_of_inertia: float = 8.333e-6,
    area: float = 0.01,
    length: float = 2.0,
) -> list[VerificationCase]:
    """The classical Euler-Bernoulli cantilever beam benchmark.

    .. code-block:: text

        Fixed
        |==============================o  Free, F (downward)
        Node 1         L                Node 2

    Analytical solution:

    .. code-block:: text

        delta = -F * L^3 / (3 * E * I)   (tip deflection, downward)
        M     = F * L                    (fixed-end moment magnitude)

    Args:
        load: Downward tip load ``F``, in newtons.
        youngs_modulus: Young's modulus, in pascals.
        moment_of_inertia: Second moment of area ``I``, in meters^4.
        area: Cross-sectional area (axial stiffness only; does not
            affect the bending results checked here), in square meters.
        length: Beam length, in meters.

    Returns:
        Two cases: tip deflection and fixed-end moment.
    """
    analytical_tip_deflection = -load * length**3 / (3.0 * youngs_modulus * moment_of_inertia)
    analytical_fixed_end_moment = load * length

    def _solve():
        material = Material(
            name="Steel", density=7850.0, youngs_modulus=youngs_modulus, poissons_ratio=0.3
        )
        section = CrossSection(area=area, second_moment_of_area=moment_of_inertia)
        node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        node_2 = Node(id=2, x=length, y=0.0, z=0.0)
        mesh = Mesh()
        mesh.add_node(node_1)
        mesh.add_node(node_2)
        mesh.add_element(
            FrameElement2D(id=1, nodes=(node_1, node_2), material=material, cross_section=section)
        )

        analysis = StaticLinearAnalysis(mesh)
        for dof in (X, Y, RZ):
            analysis.add_boundary_condition(BoundaryCondition(node_1.id, dof, 0.0))
        analysis.add_load(NodalLoad(node_2.id, Y, -load))
        return analysis.solve()

    deflection_case = VerificationCase(
        name="Cantilever beam: tip deflection",
        description="Fixed-free Euler-Bernoulli cantilever under a transverse tip load.",
        analysis_type="linear_static",
        quantity="Tip deflection",
        reference_value=analytical_tip_deflection,
        tolerance=_TIGHT_TOLERANCE,
        run=lambda: _solve().displacement(2, dof=Y),
        units="m",
        metadata={
            "load": load,
            "youngs_modulus": youngs_modulus,
            "moment_of_inertia": moment_of_inertia,
        },
    )
    moment_case = VerificationCase(
        name="Cantilever beam: fixed-end moment",
        description="Fixed-free Euler-Bernoulli cantilever under a transverse tip load.",
        analysis_type="linear_static",
        quantity="Fixed-end bending moment",
        reference_value=analytical_fixed_end_moment,
        tolerance=_TIGHT_TOLERANCE,
        run=lambda: _solve().element_bending_moment(1),
        units="N*m",
        metadata={
            "load": load,
            "youngs_modulus": youngs_modulus,
            "moment_of_inertia": moment_of_inertia,
        },
    )
    return [deflection_case, moment_case]


def thermal_conduction_cases(
    conductivity: float = 50.0,
    length: float = 2.0,
    hot_temperature: float = 373.15,
    cold_temperature: float = 293.15,
    num_elements: int = 4,
) -> list[VerificationCase]:
    """The classical steady-state 1D heat-conduction benchmark.

    For constant conductivity and no internal generation, the steady
    heat equation reduces to ``d^2T/dx^2 = 0``, whose solution is the
    linear profile:

    .. code-block:: text

        T(x) = T0 + (TL - T0) * x / L
        q    = -k * (TL - T0) / L          (heat flux, constant)

    Args:
        conductivity: Thermal conductivity ``k``, in W/(m*K).
        length: Bar length ``L``, in meters.
        hot_temperature: Prescribed temperature at ``x=0``, in kelvin.
        cold_temperature: Prescribed temperature at ``x=L``, in kelvin.
        num_elements: Number of 1D elements to mesh the bar with.

    Returns:
        Two cases: midpoint temperature and heat flux.
    """
    analytical_midpoint_temperature = hot_temperature + (cold_temperature - hot_temperature) * 0.5
    analytical_heat_flux = -conductivity * (cold_temperature - hot_temperature) / length

    def _solve():
        mechanical_material = Material(
            name="steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3
        )
        thermal_material = ThermalMaterial(
            thermal_conductivity=conductivity, density=7850.0, specific_heat=460.0
        )
        x_coords = np.linspace(0.0, length, num_elements + 1)
        nodes = [Node(id=i + 1, x=float(x), y=0.0, z=0.0) for i, x in enumerate(x_coords)]

        mesh = Mesh()
        for node in nodes:
            mesh.add_node(node)
        materials: dict[int, ThermalMaterial] = {}
        for i in range(num_elements):
            element = BarElement(
                id=i + 1,
                nodes=(nodes[i], nodes[i + 1]),
                material=mechanical_material,
                cross_section=CrossSection(area=0.01),
            )
            mesh.add_element(element)
            materials[element.id] = thermal_material

        analysis = SteadyStateThermalAnalysis(mesh, materials)
        analysis.add_boundary_condition(PrescribedTemperature(nodes[0].id, hot_temperature))
        analysis.add_boundary_condition(PrescribedTemperature(nodes[-1].id, cold_temperature))
        return analysis.solve(), nodes

    def _midpoint_temperature() -> float:
        result, nodes = _solve()
        midpoint_node = nodes[len(nodes) // 2]
        return result.node_temperature(midpoint_node.id)

    def _heat_flux() -> float:
        result, _ = _solve()
        return float(result.element_heat_flux(1)[0])

    temperature_case = VerificationCase(
        name="1D conduction: midpoint temperature",
        description="Steady-state 1D heat conduction with prescribed end temperatures.",
        analysis_type="thermal_steady_state",
        quantity="Midpoint temperature",
        reference_value=analytical_midpoint_temperature,
        tolerance=_TIGHT_TOLERANCE,
        run=_midpoint_temperature,
        units="K",
        metadata={"conductivity": conductivity, "length": length, "num_elements": num_elements},
    )
    heat_flux_case = VerificationCase(
        name="1D conduction: heat flux",
        description="Steady-state 1D heat conduction with prescribed end temperatures.",
        analysis_type="thermal_steady_state",
        quantity="Heat flux",
        reference_value=analytical_heat_flux,
        tolerance=_TIGHT_TOLERANCE,
        run=_heat_flux,
        units="W/m^2",
        metadata={"conductivity": conductivity, "length": length, "num_elements": num_elements},
    )
    return [temperature_case, heat_flux_case]


def quad_patch_case(youngs_modulus: float = 70e9, poisson_ratio: float = 0.33) -> VerificationCase:
    """The Q4 patch test: exact reproduction of a constant strain field.

    A linear displacement field ``u = a*x + b*y``, ``v = c*x + d*y`` is a
    special case of Q4's bilinear interpolation, so prescribing it on
    every node must reproduce the exact constant strain
    ``[a, d, b + c]`` at the element's center -- not an approximation,
    since Q4's shape functions can represent any linear field exactly.

    Args:
        youngs_modulus: Young's modulus, in pascals (does not affect
            the strain result, which is a pure kinematics check, but is
            required to construct a valid material).
        poisson_ratio: Poisson's ratio.

    Returns:
        One case comparing the recovered strain vector
        ``[epsilon_x, epsilon_y, gamma_xy]`` to ``[a, d, b + c]``.
    """
    a, b, c, d = 0.004, -0.002, 0.0015, 0.003
    expected_strain = np.array([a, d, b + c])
    geometry = ((0.0, 0.0), (2.0, 0.0), (2.5, 1.5), (0.3, 1.2))

    def _solve():
        material = LinearElastic2D(
            youngs_modulus=youngs_modulus, poisson_ratio=poisson_ratio, formulation="plane_stress"
        )
        nodes = [Node(id=i + 1, x=x, y=y, z=0.0) for i, (x, y) in enumerate(geometry)]
        element = QuadElement2D(id=1, nodes=tuple(nodes), material=material, thickness=0.005)
        mesh = Mesh()
        for node in nodes:
            mesh.add_node(node)
        mesh.add_element(element)

        analysis = StaticLinearAnalysis(mesh)
        for node in nodes:
            u = a * node.x + b * node.y
            v = c * node.x + d * node.y
            analysis.add_boundary_condition(BoundaryCondition(node.id, X, u))
            analysis.add_boundary_condition(BoundaryCondition(node.id, Y, v))
        result = analysis.solve()
        return np.asarray(result.element_strain(1))

    return VerificationCase(
        name="Q4 patch test: constant strain",
        description="A prescribed linear field must reproduce an exact constant strain.",
        analysis_type="linear_static",
        quantity="Element strain [eps_x, eps_y, gamma_xy]",
        reference_value=expected_strain,
        tolerance=_TIGHT_TOLERANCE,
        run=_solve,
        units="dimensionless",
    )


def cst_patch_case(youngs_modulus: float = 70e9, poisson_ratio: float = 0.33) -> VerificationCase:
    """The CST patch test: exact reproduction of a constant strain field.

    The 2D-triangle analogue of :func:`quad_patch_case`: CST's linear
    shape functions represent any linear displacement field exactly, so
    the recovered strain must match the constant-strain formula exactly
    everywhere.

    Args:
        youngs_modulus: Young's modulus, in pascals.
        poisson_ratio: Poisson's ratio.

    Returns:
        One case comparing the recovered strain vector to ``[a, d, b + c]``.
    """
    a, b, c, d = 0.004, -0.002, 0.0015, 0.003
    expected_strain = np.array([a, d, b + c])
    geometry = ((0.0, 0.0), (2.0, 0.0), (0.5, 1.5))

    def _solve():
        material = LinearElastic2D(
            youngs_modulus=youngs_modulus, poisson_ratio=poisson_ratio, formulation="plane_stress"
        )
        nodes = [Node(id=i + 1, x=x, y=y, z=0.0) for i, (x, y) in enumerate(geometry)]
        element = CSTElement2D(id=1, nodes=tuple(nodes), material=material, thickness=0.005)
        mesh = Mesh()
        for node in nodes:
            mesh.add_node(node)
        mesh.add_element(element)

        analysis = StaticLinearAnalysis(mesh)
        for node in nodes:
            u = a * node.x + b * node.y
            v = c * node.x + d * node.y
            analysis.add_boundary_condition(BoundaryCondition(node.id, X, u))
            analysis.add_boundary_condition(BoundaryCondition(node.id, Y, v))
        result = analysis.solve()
        return np.asarray(result.element_strain(1))

    return VerificationCase(
        name="CST patch test: constant strain",
        description="A prescribed linear field must reproduce an exact constant strain.",
        analysis_type="linear_static",
        quantity="Element strain [eps_x, eps_y, gamma_xy]",
        reference_value=expected_strain,
        tolerance=_TIGHT_TOLERANCE,
        run=_solve,
        units="dimensionless",
    )


def hex8_patch_case(youngs_modulus: float = 70e9, poisson_ratio: float = 0.33) -> VerificationCase:
    """The HEX8 3D patch test: exact reproduction of a constant strain field.

    The 3D analogue of :func:`quad_patch_case`. A linear nodal
    displacement field is a special case of HEX8's trilinear
    interpolation (the higher-order cross terms have zero coefficients),
    so the recovered strain must be exact, giving this toolkit's
    Version 29 3D solid benchmark (spec section 4.6): the existing HEX8
    implementation (Version 15) is mature and already carries an
    identical check in ``tests/validation/test_hex8_patch.py``.

    Args:
        youngs_modulus: Young's modulus, in pascals.
        poisson_ratio: Poisson's ratio.

    Returns:
        One case comparing the recovered strain vector
        ``[eps_x, eps_y, eps_z, gamma_xy, gamma_yz, gamma_zx]`` to its
        closed-form value.
    """
    a = (0.004, -0.002, 0.0015)
    b = (0.001, 0.003, -0.001)
    c = (-0.0005, 0.0007, 0.002)
    expected_strain = np.array([a[0], b[1], c[2], a[1] + b[0], b[2] + c[1], a[2] + c[0]])
    cube_coords = [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 1.5, 0.0),
        (0.0, 1.5, 0.0),
        (0.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
        (2.0, 1.5, 1.0),
        (0.0, 1.5, 1.0),
    ]

    def _solve():
        material = LinearElastic3D(
            youngs_modulus=youngs_modulus, poisson_ratio=poisson_ratio, density=2700.0
        )
        nodes = [Node(id=i + 1, x=x, y=y, z=z) for i, (x, y, z) in enumerate(cube_coords)]
        element = Hex8Element3D(id=1, nodes=tuple(nodes), material=material)
        mesh = Mesh()
        for node in nodes:
            mesh.add_node(node)
        mesh.add_element(element)

        analysis = StaticLinearAnalysis(mesh)
        for node in nodes:
            u = a[0] * node.x + a[1] * node.y + a[2] * node.z
            v = b[0] * node.x + b[1] * node.y + b[2] * node.z
            w = c[0] * node.x + c[1] * node.y + c[2] * node.z
            analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.X, u))
            analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.Y, v))
            analysis.add_boundary_condition(BoundaryCondition(node.id, TranslationDOF.Z, w))
        result = analysis.solve()
        return np.asarray(result.element_strain(1))

    return VerificationCase(
        name="HEX8 patch test: constant strain",
        description="A prescribed linear field must reproduce an exact constant strain in 3D.",
        analysis_type="linear_static",
        quantity="Element strain [eps_x, eps_y, eps_z, gamma_xy, gamma_yz, gamma_zx]",
        reference_value=expected_strain,
        tolerance=_TIGHT_TOLERANCE,
        run=_solve,
        units="dimensionless",
    )


def standard_benchmark_suite() -> list[VerificationCase]:
    """Return every benchmark case in this module, with its default parameters.

    Returns:
        The complete Version 29 analytical benchmark suite: axial bar,
        two-bar truss, cantilever beam, 1D thermal conduction, and
        Q4/CST/HEX8 patch tests.
    """
    return [
        *axial_bar_cases(),
        *truss_cases(),
        *cantilever_beam_cases(),
        *thermal_conduction_cases(),
        quad_patch_case(),
        cst_patch_case(),
        hex8_patch_case(),
    ]


__all__ = [
    "axial_bar_cases",
    "cantilever_beam_cases",
    "cst_patch_case",
    "hex8_patch_case",
    "quad_patch_case",
    "standard_benchmark_suite",
    "thermal_conduction_cases",
    "truss_cases",
]
