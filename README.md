# Physics-Informed Neural Networks (PINNs) for Nonlinear PDEs

This repository contains TensorFlow 1.x implementations of Physics-Informed Neural Networks (PINNs) for solving forward problems (data-driven solution) and inverse problems (data-driven parameter discovery) across four benchmark nonlinear partial differential equations.

## Implemented Solvers

| Problem | Formulation | Task | Script |
| :--- | :--- | :--- | :--- |
| **Allen-Cahn (AC)** | Discrete-time (100-stage IRK) | Forward solution | `main/allen_cahn/AC.py` |
| **Korteweg-de Vries (KdV)** | Discrete-time (50-stage IRK) | Inverse discovery (clean & 1% noise) | `main/kdv/KdV.py` |
| **Navier-Stokes (2D Cylinder Wake)** | Continuous-time | Inverse discovery & pressure inference | `main/navier_stokes/NavierStokes.py` |
| **Nonlinear Schrödinger (NLS)** | Continuous-time | Forward solution (periodic boundaries) | `main/schrodinger/Schrodinger.py` |

## Repository Structure

~~~text
.
├── README.md
├── requirements.txt
├── .gitignore
├── Utilities/
│   ├── plotting.py
│   └── IRK_weights/
│       ├── Butcher_IRK50.txt
│       └── Butcher_IRK100.txt
└── main/
    ├── Data/
    │   ├── AC.mat
    │   ├── KdV.mat
    │   ├── NLS.mat
    │   ├── cylinder_nektar_wake.mat
    │   └── cylinder_nektar_t0_vorticity.mat
    ├── allen_cahn/
    │   └── AC.py
    ├── kdv/
    │   └── KdV.py
    ├── navier_stokes/
    │   └── NavierStokes.py
    └── schrodinger/
        └── Schrodinger.py
~~~

## Environment Setup

These scripts rely on TensorFlow 1.x graph execution (`tf.Session`, `tf.placeholder`, and `tf.contrib.opt.ScipyOptimizerInterface`), which requires **Python 3.7**.

1. Create and activate a virtual environment:
~~~bash
conda create -n pinn-tf1 python=3.7
conda activate pinn-tf1
~~~

2. Install the required packages:
~~~bash
pip install -r requirements.txt
~~~

> **Note:** `Utilities/plotting.py` enables LaTeX text rendering via `pdflatex`. Ensure a LaTeX distribution (such as TeX Live or MiKTeX) is installed when generating figures, or set `"text.usetex": False` inside `Utilities/plotting.py`.

## Usage

Run each solver from its own subfolder so that the relative paths to `../Data/` and `../../Utilities/` resolve correctly:

~~~bash
# Allen-Cahn (Forward)
cd main/allen_cahn && python AC.py

# Korteweg-de Vries (Inverse)
cd main/kdv && python KdV.py

# Navier-Stokes (Inverse)
cd main/navier_stokes && python NavierStokes.py

# Nonlinear Schrödinger (Forward)
cd main/schrodinger && python Schrodinger.py
~~~

To save generated plots automatically, create a `figures/` directory inside the corresponding script folder and uncomment the `savefig(...)` call in the plotting function.
### Acknowledgements and References

This project builds upon the Physics-Informed Neural Networks (PINNs) framework introduced by Raissi, Perdikaris, and Karniadakis (2019). The publicly available reference implementation provided by the original authors was used as a starting point for the computational implementation.

The code was adapted and extended for investigating inverse PDE problems, particularly the estimation of unknown physical parameters. The original research paper and reference implementation are acknowledged as the foundational sources for this work.

**References:**

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
2. Original implementation: https://github.com/maziarraissi/PINNs
