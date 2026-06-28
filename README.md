# GeoDeepL Assignments

This repository is a collection of implementations and notes on geometric deep learning and 3D geometry processing.

It covers several core topics:

- **Mesh geometry processing**: discrete differential operators, Laplacian-based methods, diffusion on meshes, and spectral analysis on surfaces. See [`surface_operators_diffusion`](surface_operators_diffusion/).
- **Surface algorithms**: surface parameterization, registration, and related optimization-based methods. See [`surface_parameterization_registration`](surface_parameterization_registration/).
- **Neural models for 3D geometry**: PointNet, DiffusionNet, DeepSDF-style implicit models, and SIREN representations. See [`pointnet_classification_segmentation`](pointnet_classification_segmentation/), [`diffusionnet_rna_segmentation`](diffusionnet_rna_segmentation/), [`implicit_mesh_reconstruction`](implicit_mesh_reconstruction/), and [`implicit_shape_interpolation`](implicit_shape_interpolation/).
- **Applications and experiments**: practical implementations for learning, representing, and processing 3D geometric data across meshes, point clouds, and implicit surfaces.

## Repository Structure

| Folder | Assignment Goal | Main Notebook | Helper Module |
| --- | --- | --- | --- |
| [`surface_operators_diffusion`](surface_operators_diffusion/) | Discrete mesh operators, diffusion, HKS, and heat-method geodesics | `TD1_correction.ipynb` | `td1_utils.py`, `plot_utils/` |
| [`surface_parameterization_registration`](surface_parameterization_registration/) | Surface parameterization, ICP, ARAP, and non-rigid registration | `TD2_ZepengCHU.ipynb` | `td2_utils.py` |
| [`diffusionnet_rna_segmentation`](diffusionnet_rna_segmentation/) | DiffusionNet blocks and RNA mesh segmentation | `TD4_mva_geom_ZepengCHU.ipynb` | `td4_utils.py` |
| [`implicit_mesh_reconstruction`](implicit_mesh_reconstruction/) | Implicit reconstruction with SDF/DeepSDF-style networks | `TD5.ipynb` | `td5_utils.py` |
| [`implicit_shape_interpolation`](implicit_shape_interpolation/) | SIREN-based implicit shape interpolation | `TP_6_ZepengCHU.ipynb` | `tp6_utils.py` |
| [`pointnet_classification_segmentation`](pointnet_classification_segmentation/) | PointNet classification and segmentation | `Lab4.ipynb` | `lab4_utils.py` |

## Design Notes

- Notebooks remain the primary reading and execution interface.
- Local `*_utils.py` modules collect reusable model definitions, geometry operators, training helpers, and loss functions extracted from the notebooks.
- Assignment folders are independent; run notebooks from inside their own folder when using relative paths.
- Large downloaded datasets, checkpoints, and generated outputs should stay local and are ignored by default.
- `surface_operators_diffusion` includes small required mesh assets under `surface_operators_diffusion/data/meshes/`.

## Setup

Create an environment and install the shared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some notebooks still include Colab-style setup cells that download course datasets or helper archives.
# MVA-Geometry-Processing-and-Geometric-Deep-Learning
