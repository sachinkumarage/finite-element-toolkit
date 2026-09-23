"""Example: verifying 1D steady-state heat conduction against its analytical solution (Version 29).

**Engineering problem.** A bar with constant thermal conductivity has
its two ends held at fixed temperatures, with no internal heat
generation.

**Analytical solution.** The steady heat equation reduces to
``d^2T/dx^2 = 0``, giving a linear temperature profile:

.. code-block:: text

    T(x) = T0 + (TL - T0) * x / L
    q    = -k * (TL - T0) / L      (constant heat flux)

**FEA model.** A chain of 1D thermal conduction elements (reusing
:class:`~femtoolkit.mesh.bar_element.BarElement`'s geometry) -- exact
for a linear temperature profile.

**Verification procedure.** :func:`~femtoolkit.verification.benchmarks.thermal_conduction_cases`
builds the midpoint-temperature and heat-flux cases; the runner reports
error against a tight tolerance.
"""

from __future__ import annotations

from femtoolkit.verification.benchmarks import thermal_conduction_cases
from femtoolkit.verification.runner import VerificationRunner


def main() -> None:
    """Verify a 1D steady-state conduction bar's midpoint temperature and heat flux."""
    print("Finite Element Toolkit")
    print("Version 29 -- 1D Thermal Conduction Verification")
    print("=" * 50)

    conductivity, length, hot, cold = 50.0, 2.0, 373.15, 293.15
    cases = thermal_conduction_cases(
        conductivity=conductivity, length=length, hot_temperature=hot, cold_temperature=cold
    )

    print(f"\nk = {conductivity} W/(m*K), L = {length} m, T0 = {hot} K, TL = {cold} K")

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
