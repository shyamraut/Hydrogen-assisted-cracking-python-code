# Hydrogen-Assisted Cracking (HAC) Analysis Framework

This repository contains a Python framework for simulating and analyzing Hydrogen-Assisted Cracking (HAC) in materials using Finite Element Analysis (FEA). The framework solves for displacements, phase fields (fracture), and hydrogen concentration using a combination of numerical methods, including the Newton-Raphson method and sparse matrix solvers.

## Features

- **Finite Element Analysis (FEA)**: Solves for displacement, strain, and stress across a material mesh.
- **Phase Field Simulation**: Computes the phase field for crack propagation.
- **Hydrogen Concentration Simulation**: Models hydrogen diffusion and its effect on material properties.
- **Visualization**: Generates contour plots for displacement, phase field, and hydrogen concentration.
- **Export to VTK**: Outputs results to `.vtu` format for visualization in Paraview or other VTK-compatible tools.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Input Files](#input-files)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed along with the following dependencies:

- `numpy`
- `matplotlib`
- `scipy`
- `termcolor`

To install the required Python packages, run:

```bash
pip install -r requirements.txt
```

Alternatively, you can manually install the dependencies:

```bash
pip install numpy matplotlib scipy termcolor
```

### Optional

- **ParaView**: To visualize `.vtu` files generated by the code, download and install [ParaView](https://www.paraview.org/).

## Usage

### Running the Simulation

1. Place the input file (e.g., `SENPshear.inp`) in the project directory.
2. Run the script using Python:

   ```bash
   python main.py
   ```

3. The program will output contour plots for displacement, phase field, and hydrogen concentration during each load step. These will be saved in their respective directories.

4. Additionally, VTK files for each step will be saved in the `VTK_files` directory, which can be visualized using ParaView.

### Example Workflow

```bash
python main.py
```

You will see outputs like:

```bash
Load step 1 - Iteration 1 - tolerance = 0.00123
Phase field solve step 1 - Iteration 1 - tolerance = 0.00234
Plot saved: displacement_Contour_Plot_step_1.png
VTK file written: out-1.vtu
...
```

### Inputs

The primary input is an Abaqus `.inp` file containing information about the mesh (nodes and elements). The framework expects the input file to contain:
- Node definitions
- Element connectivity
- Boundary condition sets

Make sure the input file follows the expected format (see the sample input file in the repository).

## Project Structure

```
├── input_parser.py       # Functions for parsing nodes and elements from Abaqus .inp files
├── matrix_computation.py # Functions for assembling stiffness matrices, forces, and solving equations
├── plotting.py           # Functions to create contour plots for displacement, phase field, and concentration
├── main.py               # Main driver script for the simulation
├── requirements.txt      # List of required Python packages
├── VTK_files/            # Directory to store VTK output files
├── displacement_Contour_Plot/  # Directory to store displacement plots
├── Phase_Field_Contour_plot/   # Directory to store phase field plots
├── hydrogen_concentration_contour_plot/  # Directory to store hydrogen concentration plots
└── README.md             # This README file
```

## Visualization

1. **Displacement Contour Plot**: Shows the magnitude of displacement at each node after each load step.
   
   ![Displacement Contour Plot](displacement_Contour_Plot_example.png)

2. **Phase Field Contour Plot**: Visualizes the phase field (crack propagation) during the simulation.
   
   ![Phase Field Contour Plot](Phase_Field_Contour_Plot_example.png)

3. **Hydrogen Concentration Contour Plot**: Displays the distribution of hydrogen concentration across the material.
   
   ![Hydrogen Concentration Contour Plot](hydrogen_concentration_contour_plot_example.png)

4. **VTK Output**: Files in `.vtu` format are saved in the `VTK_files/` folder for post-processing in ParaView.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.


---
