# Finite Element Toolkit

An open-source Python toolkit for developing finite element analysis (FEA) capabilities, built incrementally as a series of versioned milestones.

**This is the Version 1 foundation release.** It establishes the project's architecture and core domain model — materials, nodes, elements, and a mesh container — but it does **not** yet contain a finite element solver. No stiffness matrices, matrix assembly, boundary condition solving, or stress/strain calculations are implemented in this version.

## Current Features

- **Material** — data model for isotropic material properties (density, Young's modulus, Poisson's ratio)
- **Node** — a point in 3D space, identified by ID and coordinates
- **Element** — connects nodes through a material, with no computation performed
- **Mesh** — a container managing nodes and elements with referential-integrity validation
- **Engineering units foundation** — named SI unit constants
- **Custom exceptions** — domain-specific error types
- **Logging** — a package logger that stays silent unless the host application configures it
- **Tests** — a pytest suite covering the domain model

## Installation

Clone the repository and install it in editable mode:

```bash
git clone <repository-url>
cd finite-element-toolkit
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```python
from femtoolkit.materials import Material
from femtoolkit.mesh import Element, Mesh, Node

steel = Material(name="Steel", density=7850.0, youngs_modulus=200e9, poissons_ratio=0.3)

node_1 = Node(id=1, x=0.0, y=0.0, z=0.0)
node_2 = Node(id=2, x=1.0, y=0.0, z=0.0)
element = Element(id=1, nodes=[node_1, node_2], material=steel)

mesh = Mesh()
mesh.add_node(node_1)
mesh.add_node(node_2)
mesh.add_element(element)

print(mesh.get_node(1))
print(mesh.get_element(1))
```

## Project Structure

```text
finite-element-toolkit/
├── src/femtoolkit/
│   ├── materials/       # Material data model
│   ├── mesh/             # Node, Element, and Mesh
│   ├── units/             # SI unit constants
│   ├── exceptions/         # Custom exception types
│   ├── config.py            # Package metadata and defaults
│   └── logging_config.py    # Package logger configuration
├── examples/              # Runnable example scripts
└── tests/                  # pytest test suite
```

## Testing

Run the test suite with pytest:

```bash
pytest
```

## Code Quality

Check code style and lint rules with Ruff:

```bash
ruff check .
```

## Example

Run the example script to build and print a minimal model:

```bash
python examples/create_basic_model.py
```

## Roadmap

Future versions will build the FEA solver on top of this foundation. None of the following is implemented yet:

- **Version 2** — Basic FEA mathematics and degrees of freedom
- **Version 3** — 1D bar/truss fundamentals
- **Version 4** — Beam analysis
- **Version 5** — 2D finite elements
- **Version 6** — Advanced mesh and visualization
- **Later** — 3D elements, GUI, reporting, and more

## License

Released under the [MIT License](LICENSE).
