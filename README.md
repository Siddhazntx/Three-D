# 3D Reconstruction using Depth Anything V3

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-EE4C2C?logo=pytorch)
![Open3D](https://img.shields.io/badge/Open3D-%3E%3D0.18-success)
![License](https://img.shields.io/badge/License-MIT-yellow)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia)

**Monocular 3D Reconstruction · Point Cloud · Segmentation · Voxel Mesh · Gaussian Splatting**

</div>

---

## Overview

This repository implements a full **Monocular 3D Reconstruction pipeline** powered by the **Depth Anything V3 (DA3)** foundation model. Starting from a folder of standard RGB images — no LiDAR, no stereo rig — the system produces:

- Metric depth maps with per-pixel confidence scores
- Registered, texture-mapped 3D point clouds (`.ply`)
- Ground-plane and object segmentation labels
- Voxel cube meshes (RGB + segmentation-colored)
- Gaussian Splatting exports (`.ply` splat clouds + `.glb` scenes)
- Compressed NumPy archives (`.npz`) for downstream pipelines

The pipeline is implemented as a single, heavily-commented script (`da_3d_reconstruction.py`, ~900 lines, 14 modular steps) and is designed to run top-to-bottom in a Jupyter-style cell workflow.

---

## Key Features

| Feature | Details |
|---|---|
| **Zero-Shot Depth Estimation** | DA3 `NESTED-GIANT-LARGE` ViT backbone, no scene-specific training |
| **Metric + Relative Depth** | Predicts both absolute scale and relative structure |
| **Confidence Filtering** | Per-pixel confidence map used to prune unreliable points |
| **Multi-Frame ICP Registration** | Point-to-plane ICP with auto-scale voxel downsampling |
| **Statistical Outlier Removal** | Scipy `cKDTree`-based SOR, parallelized with `workers=-1` |
| **RANSAC Plane Segmentation** | Vectorized NumPy RANSAC + SVD refinement, multi-plane |
| **Euclidean Object Clustering** | Voxel-graph connected components via `scipy.sparse` |
| **KNN Boundary Refinement** | Majority-vote label smoothing on cluster boundaries |
| **Voxel Mesh Generation** | Fully vectorized cube-mesh builder, no Python loops |
| **Gaussian Splatting Export** | DA3 native GS PLY + GLB scene with camera frustums |
| **Interactive ROI Crop** | Open3D `VisualizerWithEditing` for bounding-box selection |

---

## Pipeline — 14 Steps

```
Input Images (RGB folder)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1 ── Load DA3 Model  (CUDA / CPU auto-detect)     │
│  Step 2 ── Scan Image Folder (jpg / png / jpeg)         │
│  Step 3 ── DA3 Inference                                │
│             ├─ Depth maps    [N × H × W]                │
│             ├─ Confidence    [N × H × W]                │
│             ├─ Intrinsics K  [N × 3 × 3]                │
│             └─ Extrinsics    [N × 4 × 4]  (w2c)        │
│  Step 4 ── Back-projection → 3D Point Cloud             │
│             Z = depth                                   │
│             X = (u − cx) × Z / fx                      │
│             Y = (v − cy) × Z / fy                      │
│  Step 4b─ Merge All Frames (confidence-filtered)        │
│  Step 5 ── Statistical Outlier Removal (SOR)            │
│  Step 6 ── Interactive ROI Selection (optional)         │
│  Step 7 ── Two-Frame Registration Preview (ICP)         │
│  Step 8 ── Full Multi-Frame ICP Registration            │
│  Step 9 ── Multi-Plane Segmentation (RANSAC + SVD)      │
│  Step 10── Object Clustering (voxel graph)              │
│  Step 11── Label Merge + KNN Boundary Refinement        │
│  Step 12── Voxel Mesh Generation                        │
│  Step 13── Gaussian Splatting + GLB Export              │
│  Step 14── Save All Outputs                             │
└─────────────────────────────────────────────────────────┘
        │
        ▼
results/<SCENE>/
  ├── reconstruction.ply          ← RGB point cloud + seg labels
  ├── voxel_mesh_rgb.ply          ← Voxel cube mesh (texture)
  ├── voxel_mesh_seg.ply          ← Voxel cube mesh (seg colors)
  ├── reconstruction_data.npz     ← All arrays for downstream use
  ├── gs_ply/                     ← Gaussian Splatting splat clouds
  └── <scene>.glb                 ← GLB scene (mesh + cameras)
```

---

## Technical Details

### Depth Estimation
The DA3 model (`depth-anything/DA3NESTED-GIANT-LARGE`) is loaded from HuggingFace and run in `infer_gs=True` mode, which simultaneously predicts depth, confidence, per-frame intrinsics, and multi-frame extrinsics — enabling a complete SfM-style output from a single inference call.

### Back-Projection
Using the predicted pinhole camera intrinsics:

$$Z = d, \quad X = \frac{(u - c_x)\,Z}{f_x}, \quad Y = \frac{(v - c_y)\,Z}{f_y}$$

Points are then transformed from camera to world space using the predicted `w2c` extrinsics (inverse rigid transform).

### ICP Registration
Each frame's center-zone, high-confidence points are voxel-downsampled and aligned to frame 0 via **point-to-plane ICP**. A max-translation guard (`max_shift_pct=0.05` of bounding-box diagonal) prevents runaway corrections; frames exceeding this threshold fall back to identity.

### Plane Segmentation
Iterative NumPy RANSAC (`n_iterations` random 3-point trials, vectorized scoring in batches of 256) followed by SVD-based least-squares refinement on the inlier set. Supports arbitrary `n_planes`; each detected plane is removed before the next iteration.

### Object Clustering
Points above detected planes are assigned to a 3D voxel grid. A 26-connected voxel adjacency graph is built via `numpy.searchsorted` and solved with `scipy.sparse.csgraph.connected_components`. This runs in milliseconds on millions of points and has no stochastic variance.

### Voxel Mesh
Each occupied voxel becomes an axis-aligned unit cube. Vertices and triangles for all voxels are generated in a single NumPy broadcast — no per-cube Python loops.

---

## Repository Structure

```
3d-Reconstruction/
├── data/
│   └── <SCENE>/            # Input RGB images (.jpg / .png)
├── results/
│   └── <SCENE>/
│       ├── masks/           # Intermediate mask files
│       ├── depth_vis/       # Depth map visualizations
│       ├── gs_ply/          # Gaussian Splatting PLY per frame
│       ├── reconstruction.ply
│       ├── voxel_mesh_rgb.ply
│       ├── voxel_mesh_seg.ply
│       ├── reconstruction_data.npz
│       └── <scene>.glb
├── assets/                  # README images and demo GIFs
├── da_3d_reconstruction.py  # Main 14-step pipeline script
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Siddhazntx/3d-Reconstruction.git
cd 3d-Reconstruction
```

### 2. Install Depth Anything V3

> ⚠️ This must be done **before** installing `requirements.txt`.

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V3.git
cd Depth-Anything-V3
pip install -e .
cd ..
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
torch>=2.0
numpy<2
open3d>=0.18
scipy>=1.10
matplotlib>=3.7
```

### 4. (Optional) GPU Setup

A CUDA-capable GPU is strongly recommended for DA3 inference. The script auto-detects CUDA:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

For CPU-only runs, inference will be significantly slower on high-resolution images.

---

## Usage

### 1. Prepare Your Scene

Place your input images in `data/<YOUR_SCENE>/`:

```
data/
└── MY_SCENE/
    ├── frame_001.jpg
    ├── frame_002.jpg
    └── ...
```

### 2. Set the Scene Name

At the top of `da_3d_reconstruction.py`, set:

```python
SCENE = "MY_SCENE"
```

### 3. Run the Pipeline

```bash
python da_3d_reconstruction.py
```

The script runs all 14 steps sequentially. Each step prints progress to console and opens an interactive Open3D viewer window where applicable.

### 4. Interactive ROI (Step 6)

When the ROI viewer opens:
1. Press **`K`** to lock the viewpoint
2. **Click and drag** to draw a selection rectangle
3. Press **`C`** to crop
4. Press **`Q`** to close

If you skip (close without cropping), the full point cloud is used for registration.

---

## Output Files

| File | Format | Contents |
|---|---|---|
| `reconstruction.ply` | Binary PLY | XYZ + RGB + `seg_label` + `is_ground` per point |
| `voxel_mesh_rgb.ply` | PLY mesh | Voxel cube mesh with original texture colors |
| `voxel_mesh_seg.ply` | PLY mesh | Voxel cube mesh colored by segmentation label |
| `reconstruction_data.npz` | NumPy NPZ | `depth`, `conf`, `intrinsics`, `extrinsics`, `processed_images`, `points_3d`, `colors_3d` |
| `gs_ply/` | PLY (GS) | Per-frame Gaussian Splatting splat clouds |
| `<scene>.glb` | Binary GLTF | Full scene mesh + RGB textures + camera frustums |

> **Note:** `.glb` and high-resolution `.ply` files are stored locally due to file size. Sample assets are in `results/Sample_img/`.

---

## Results

| Input Image (2D) | Depth Map | Reconstructed Point Cloud (3D) |
|---|---|---|
| [Sample Input](data/Sample_img/image1.jpg) | [Depth Vis](results/Sample_img/depth_vis/0000.jpg) | Stored locally as `.ply` / `.glb` |

The depth map shows the spatial consistency achieved by the DA3 backbone. Point clouds are viewable in **Open3D**, **MeshLab**, or **Blender** (import `.ply`). GLB files can be dragged directly into any web browser or **three.js** viewer.

---

## Key Parameters

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `SCENE` | top of script | `"MY_SCENE"` | Sets input/output folder |
| `conf_thresh` | Steps 4, 8 | `0.4 – 0.9` | Confidence gate for point inclusion |
| `std_ratio` | Step 5 | `1.0` | SOR aggressiveness (lower = more aggressive) |
| `nb_neighbors` | Step 5 | `20` | Neighbour count for SOR |
| `n_planes` | Step 9 | `2` | Number of planes to detect |
| `distance_thresh` | Step 9 | `0.01` | RANSAC inlier threshold (scene units) |
| `voxel_size` | Step 12 | auto | Voxel edge length; auto-computed from point density |
| `target_voxels` | Step 12 | `200,000` | Budget for auto voxel size calculation |

---

## Visualization Tools

The pipeline produces Open3D interactive windows at multiple stages. You can also load any output `.ply` externally:

**Open3D (Python)**
```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("results/MY_SCENE/reconstruction.ply")
o3d.visualization.draw_geometries([pcd])
```

**MeshLab** — `File → Import Mesh → reconstruction.ply`

**Blender** — `File → Import → Stanford PLY`

**Browser (GLB)** — Drag `<scene>.glb` into [glTF Viewer](https://gltf-viewer.donmccurdy.com/)

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0 | DA3 model inference |
| `depth-anything-3` | latest | Depth + pose estimation |
| `open3d` | ≥ 0.18 | Point cloud I/O, ICP, visualization, mesh export |
| `numpy` | < 2 | All array operations |
| `scipy` | ≥ 1.10 | KDTree SOR, sparse graph clustering |
| `matplotlib` | ≥ 3.7 | Depth/confidence map visualization |

---

## Roadmap

- [ ] Streamlit / Gradio web interface for drag-and-drop reconstruction
- [ ] Batch processing mode (multiple scenes in one run)
- [ ] Poisson surface reconstruction from cleaned point cloud
- [ ] Support for video input (frame extraction + deduplication)
- [ ] Docker container with all dependencies pre-installed
- [ ] Optional metric scale calibration via known object size

---

## Citation

If you use this project or the Depth Anything V3 backbone in your work, please cite the original DA3 paper:

```bibtex
@article{depth_anything_v3,
  title   = {Depth Anything V3},
  author  = {Depth Anything Team},
  journal = {arXiv},
  year    = {2025}
}
```

---

## Acknowledgements

- [Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3) — Foundation model for monocular depth and pose estimation
- [Open3D](http://www.open3d.org/) — 3D data processing and visualization
- [SciPy](https://scipy.org/) — KDTree and sparse graph algorithms

---

<div align="center">
Made with ❤️ · <a href="https://github.com/Siddhazntx/3d-Reconstruction">github.com/Siddhazntx/3d-Reconstruction</a>
</div>
