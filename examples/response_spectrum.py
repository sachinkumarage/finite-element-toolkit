"""Example: response spectrum representation and modal spectral response, Version 12.

Demonstrates :class:`~femtoolkit.analysis.spectrum.ResponseSpectrum`
(interpolated spectral acceleration vs. period) and
:func:`~femtoolkit.analysis.spectrum.modal_spectral_response` (per-mode
peak response to that spectrum) on a cantilevered Q4 plate -- the
mathematical foundation this version provides, deliberately stopping
short of any seismic-code-specific workflow or modal-combination rule
(see the module's own docstring).
"""

from femtoolkit.analysis import BoundaryCondition, TranslationDOF
from femtoolkit.analysis.dynamic_system import build_dynamic_system
from femtoolkit.analysis.modal import modal_analysis_of_system
from femtoolkit.analysis.spectrum import ResponseSpectrum, modal_spectral_response
from femtoolkit.geometry import Rectangle
from femtoolkit.materials import LinearElastic2D
from femtoolkit.mesh import create_quad_mesh

YOUNGS_MODULUS = 200e9  # Pa
POISSON_RATIO = 0.3
DENSITY = 7850.0  # kg/m^3
THICKNESS = 0.01  # m
WIDTH = 0.5  # m
HEIGHT = 0.1  # m
NX = 10
NY = 3
NUM_MODES = 5


def main() -> None:
    """Build a simple design-style spectrum and evaluate a plate's modal response to it."""
    # A simple, illustrative acceleration response spectrum (not any
    # particular building code's design spectrum): rises to a plateau,
    # then decays -- the typical shape of a real spectrum. Values are in
    # m/s^2 (SI), matching every other unit in this toolkit --
    # ResponseSpectrum itself performs no unit conversion, so mixing
    # unit systems (e.g. "g") between the spectrum and the modal
    # properties it is evaluated against would silently corrupt the
    # q_max = Gamma*Sa/omega^2 formula's units.
    peak_g = 1.0  # peak spectral acceleration, in units of g
    gravity = 9.81  # m/s^2
    spectrum = ResponseSpectrum(
        periods=[0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        accelerations=[
            v * peak_g * gravity for v in (0.4, 1.0, 1.0, 0.6, 0.3, 0.15, 0.05)
        ],
    )

    print("Response spectrum interpolation:")
    for period in (0.0, 0.02, 0.075, 0.3, 1.5, 5.0):
        print(f"    Sa({period:.3f} s) = {spectrum.evaluate(period):.4f} m/s^2")

    material = LinearElastic2D(
        youngs_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        formulation="plane_stress",
        density=DENSITY,
    )
    domain = Rectangle(width=WIDTH, height=HEIGHT)
    mesh = create_quad_mesh(
        width=WIDTH, height=HEIGHT, nx=NX, ny=NY, material=material, thickness=THICKNESS
    )

    boundary_conditions = []
    for node in mesh.nodes_on_boundary(domain.boundary("left")):
        boundary_conditions.append(BoundaryCondition(node.id, TranslationDOF.X, 0.0))
        boundary_conditions.append(BoundaryCondition(node.id, TranslationDOF.Y, 0.0))

    system = build_dynamic_system(mesh, boundary_conditions)
    modal_result = modal_analysis_of_system(system, num_modes=NUM_MODES, direction="y")

    response = modal_spectral_response(modal_result, spectrum)

    print_summary(response)


def print_summary(response) -> None:
    """Print the per-mode spectral response quantities."""
    print("\nFinite Element Toolkit")
    print("Version 12 -- Response Spectrum and Modal Spectral Response")
    print("=" * 66)

    header = f"    {'Mode':<6}{'T (s)':>10}{'Sa (m/s2)':>12}{'q_max':>14}{'F_eq (N)':>14}"
    print(header)
    for i in range(len(response.periods)):
        print(
            f"    {i + 1:<6}{response.periods[i]:>10.5f}"
            f"{response.spectral_accelerations[i]:>12.4f}"
            f"{response.modal_displacements[i]:>14.6e}"
            f"{response.equivalent_static_forces[i]:>14.4f}"
        )

    print("\nEach row is an independent per-mode peak response (q_max, F_eq); combining")
    print("them into one estimated total (e.g. via SRSS) is out of scope for this version.")


if __name__ == "__main__":
    main()
