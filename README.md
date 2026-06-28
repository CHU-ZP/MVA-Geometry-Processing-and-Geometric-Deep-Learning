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
| [`surface_operators_diffusion`](surface_operators_diffusion/) | Discrete mesh operators, diffusion, HKS, and heat-method geodesics | `mesh_operators_diffusion.ipynb` | `td1_utils.py`, `plot_utils/` |
| [`surface_parameterization_registration`](surface_parameterization_registration/) | Surface parameterization, ICP, ARAP, and non-rigid registration | `surface_parameterization_registration.ipynb` | `td2_utils.py` |
| [`diffusionnet_rna_segmentation`](diffusionnet_rna_segmentation/) | DiffusionNet blocks and RNA mesh segmentation | `diffusionnet_rna_segmentation.ipynb` | `td4_utils.py` |
| [`pointnet_classification_segmentation`](pointnet_classification_segmentation/) | PointNet classification and segmentation | `pointnet_classification_segmentation.ipynb` | `lab4_utils.py` |
| [`implicit_mesh_reconstruction`](implicit_mesh_reconstruction/) | Implicit reconstruction with SDF/DeepSDF-style networks | `implicit_mesh_reconstruction.ipynb` | `td5_utils.py` |
| [`implicit_shape_interpolation`](implicit_shape_interpolation/) | SIREN-based implicit shape interpolation | `implicit_shape_interpolation.ipynb` | `tp6_utils.py` |

## Assignment Overview

### Mesh Operators and Diffusion

**Objective.** Build the basic discrete differential geometry pipeline for triangle meshes, then use Laplacian-based operators for diffusion, spectral analysis, and geometric descriptors.

**Core ideas.**

- OFF mesh loading and lightweight mesh containers
- Face areas, vertex area matrices, and cotangent weights
- Cotangent Laplacian / stiffness matrix construction
- Full and spectral heat diffusion on surfaces
- Heat Kernel Signature (HKS)
- Heat-method geodesic distance approximation

**Implementation.** [`surface_operators_diffusion`](surface_operators_diffusion/) contains the completed notebook and mesh assets. `td1_utils.py` factors out the mesh IO, discrete operators, diffusion routines, HKS computation, and heat-method helpers.

### Surface Parameterization and Registration

**Objective.** Implement classical surface processing algorithms for flattening, aligning, and deforming meshes with optimization-based methods.

**Core ideas.**

- Boundary edge extraction and ordered boundary loops
- Tutte embedding with fixed circular boundary constraints
- Least Squares Conformal Maps (LSCM)
- Nearest-neighbor correspondence search
- Rigid ICP with Procrustes alignment
- ARAP deformation and non-rigid ICP-style registration

**Implementation.** [`surface_parameterization_registration`](surface_parameterization_registration/) contains the parameterization and registration notebook. `td2_utils.py` collects the reusable boundary, parameterization, ICP, and ARAP utilities extracted from the notebook.

### DiffusionNet RNA Segmentation

**Objective.** Implement the main components of DiffusionNet and apply them to vertex-wise segmentation on RNA mesh data.

**Core ideas.**

- Projection and unprojection between vertex functions and spectral bases
- Learnable spectral diffusion layers
- Spatial gradient feature modules
- Residual DiffusionNet blocks with MLP updates
- HKS/WKS input features
- Segmentation training, validation, visualization, and ablation studies

**Implementation.** [`diffusionnet_rna_segmentation`](diffusionnet_rna_segmentation/) contains the DiffusionNet notebook. `td4_utils.py` gathers the spectral diffusion modules, network blocks, trainer logic, and feature utilities.

### PointNet Classification and Segmentation

**Objective.** Study point-cloud learning with PointNet, including classification on ModelNet-style data and segmentation on mesh-derived point samples.

**Core ideas.**

- PointNet feature transforms and shared point-wise MLPs
- Global max-pooling for permutation-invariant shape representations
- Classification and segmentation heads
- Point-cloud data augmentation with scaling, translation, and rotation
- ModelNet40 data loading from HDF5 files
- Regularized losses and training loops

**Implementation.** [`pointnet_classification_segmentation`](pointnet_classification_segmentation/) contains the PointNet notebook. `lab4_utils.py` factors out the PointNet modules, datasets, losses, and trainer classes.

### Implicit Mesh Reconstruction

**Objective.** Reconstruct surfaces from point clouds using signed and unsigned implicit distance representations.

**Core ideas.**

- Signed distance estimation from oriented point clouds
- Grid-based SDF evaluation and marching cubes reconstruction
- DeepSDF-style neural implicit functions
- Eikonal-style regularization
- Unsigned distance and SAL-style losses
- Mesh export and qualitative reconstruction comparison

**Implementation.** [`implicit_mesh_reconstruction`](implicit_mesh_reconstruction/) contains the reconstruction notebook. `td5_utils.py` collects SDF computation, neural field definitions, loss functions, training loops, and reconstruction helpers.

### Implicit Shape Interpolation

**Objective.** Use SIREN implicit neural representations to model and interpolate geometric shapes.

**Core ideas.**

- Sinusoidal representation networks (SIREN)
- Shape data attachment and SDF supervision
- Eikonal regularization
- Level-set and vector-field constraints
- Morphing between synthetic shapes
- Implicit shape visualization through marching squares

**Implementation.** [`implicit_shape_interpolation`](implicit_shape_interpolation/) contains the SIREN interpolation notebook. `tp6_utils.py` gathers the SIREN layers, losses, optimization routines, vector-field helpers, and morphing utilities.

## Design Notes

- Notebooks remain the primary reading and execution interface.
- Local `*_utils.py` modules collect reusable model definitions, geometry operators, training helpers, and loss functions extracted from the notebooks.
- Assignment folders are independent; run notebooks from inside their own folder when using relative paths.
- Large downloaded datasets, checkpoints, and generated outputs should stay local and are ignored by default.
- `surface_operators_diffusion` includes small required mesh assets under `surface_operators_diffusion/data/meshes/`.
