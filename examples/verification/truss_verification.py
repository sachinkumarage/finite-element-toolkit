"""Example: verifying a symmetric two-bar truss against its analytical solution (Version 29).

**Engineering problem.** Two truss members meet at a free apex and are
pinned at two ground supports; a vertical load ``F`` is applied at the
apex.

**Analytical solution.** Solved by joint equilibrium at the apex alone
(two members, two equilibrium equations) plus the unit-load method for
displacement -- see :func:`~femtoolkit.verification.benchmarks.truss_cases`'s
docstring for the full derivation:

.. code-block:: text

    N     = -F * L / (2 * height)
    delta = F * L^3 / (2 * height^2 * A * E)

**FEA model.** Two :class:`~femtoolkit.mesh.truss_element.TrussElement2D`
elements sharing the apex node.

**Verification procedure.** :func:`~femtoolkit.verification.benchmarks.truss_cases`
builds the displacement and member-force cases; the runner reports
error against a tight tolerance.
"""

from __future__ import annotations

from femtoolkit.verification.benchmarks import truss_cases
from femtoolkit.verification.runner import VerificationRunner


def main() -> None:
    """Verify a symmetric two-bar truss's apex displacement and member force."""
    print("Finite Element Toolkit")
    print("Version 29 -- Two-Bar Truss Verification")
    print("=" * 42)

    load, area, youngs_modulus, half_span, height = 1000.0, 0.001, 200e9, 1.0, 1.0
    cases = truss_cases(
        load=load, area=area, youngs_modulus=youngs_modulus, half_span=half_span, height=height
    )

    print(f"\nApex load F = {load} N, half-span = {half_span} m, height = {height} m")

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
