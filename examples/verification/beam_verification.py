"""Example: verifying an Euler-Bernoulli cantilever beam (Version 29).

**Engineering problem.** A cantilever beam, fixed at one end, carries a
transverse point load ``F`` at the free tip.

**Analytical solution** (Euler-Bernoulli beam theory):

.. code-block:: text

    delta = -F * L^3 / (3 * E * I)   (tip deflection, downward)
    M     = F * L                    (fixed-end moment magnitude)

**FEA model.** A single :class:`~femtoolkit.mesh.frame_element.FrameElement2D`
-- exact for a prismatic Euler-Bernoulli beam under this loading.

**Verification procedure.** :func:`~femtoolkit.verification.benchmarks.cantilever_beam_cases`
builds the deflection and moment cases; the runner reports error
against a tight tolerance.
"""

from __future__ import annotations

from femtoolkit.verification.benchmarks import cantilever_beam_cases
from femtoolkit.verification.runner import VerificationRunner


def main() -> None:
    """Verify a cantilever beam's tip deflection and fixed-end moment against theory."""
    print("Finite Element Toolkit")
    print("Version 29 -- Cantilever Beam Verification")
    print("=" * 44)

    load, youngs_modulus, moment_of_inertia, length = 1000.0, 200e9, 8.333e-6, 2.0
    cases = cantilever_beam_cases(
        load=load, youngs_modulus=youngs_modulus, moment_of_inertia=moment_of_inertia, length=length
    )

    print(f"\nTip load F = {load} N, E = {youngs_modulus:.3e} Pa, I = {moment_of_inertia:.3e} m^4")

    report = VerificationRunner().run_all(cases)
    for result in report.results:
        print(f"\n{result.case_name}")
        print(f"  Analytical: {result.reference_value:.6e} {result.units}")
        print(f"  FEA:        {result.numerical_value:.6e} {result.units}")
        print(f"  Relative error: {result.relative_error:.3e}")
        print(f"  Status: {result.status.value.upper()}")

    print(f"\n{report.passed}/{len(report.results)} cases passed.")


if __name__ == "__main__":
    main()
