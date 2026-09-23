"""Example: verifying a fixed-free axial bar against its analytical solution (Version 29).

**Engineering problem.** A steel bar, fixed at one end, is pulled by an
axial force ``F`` at the free end.

**Analytical solution.** For a uniform bar (constant ``E``, ``A``, ``L``):

.. code-block:: text

    delta = F * L / (A * E)     (tip displacement)
    sigma = F / A                (axial stress)

**FEA model.** A single :class:`~femtoolkit.mesh.bar_element.BarElement`
-- since a bar element's stiffness is exact for uniform axial strain,
the FEA result should match the analytical solution to floating-point
precision, not just approximately.

**Verification procedure.** :func:`~femtoolkit.verification.benchmarks.axial_bar_cases`
builds two :class:`~femtoolkit.verification.cases.VerificationCase`
instances (displacement, stress); :class:`~femtoolkit.verification.runner.VerificationRunner`
runs them and reports the error against a tight
(:data:`~femtoolkit.verification.tolerance.Tolerance`) tolerance.
"""

from __future__ import annotations

from femtoolkit.verification.benchmarks import axial_bar_cases
from femtoolkit.verification.runner import VerificationRunner


def main() -> None:
    """Verify a fixed-free axial bar's displacement and stress against theory."""
    print("Finite Element Toolkit")
    print("Version 29 -- Axial Bar Verification")
    print("=" * 40)

    load, area, youngs_modulus, length = 1000.0, 0.01, 200e9, 2.0
    cases = axial_bar_cases(load=load, area=area, youngs_modulus=youngs_modulus, length=length)

    print(f"\nLoad F = {load} N, Area A = {area} m^2, E = {youngs_modulus:.3e} Pa, L = {length} m")

    report = VerificationRunner().run_all(cases)
    for result in report.results:
        print(f"\n{result.case_name}")
        print(f"  Quantity:         {result.quantity}")
        print(f"  Analytical value: {result.reference_value:.6e} {result.units}")
        print(f"  FEA value:        {result.numerical_value:.6e} {result.units}")
        print(f"  Relative error:   {result.relative_error:.3e}")
        print(
            f"  Tolerance:        rel={result.tolerance.relative:.1e}, "
            f"abs={result.tolerance.absolute:.1e}"
        )
        print(f"  Status:           {result.status.value.upper()}")

    print(f"\n{report.passed}/{len(report.results)} cases passed.")


if __name__ == "__main__":
    main()
