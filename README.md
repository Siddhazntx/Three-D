# 3D Reconstruction using Depth Anything V3

This repository implements a high-fidelity **Monocular 3D Reconstruction** pipeline. By leveraging the **Depth Anything V3** foundation model, the system transforms standard 2D images into spatial 3D point clouds without the need for specialized hardware like LiDAR or stereo cameras.

## 🚀 Overview
The core logic involves extracting a high-resolution depth map from a single RGB image and using geometric back-projection (Pinhole Camera Model) to map pixels into 3D space ($X, Y, Z$).

### Key Features
* **Zero-Shot Depth Estimation:** Powered by Depth Anything V3 (ViT-based architecture).
* **3D Point Cloud Generation:** Converts depth maps to `.ply` files for visualization.
* **RGB Mapping:** Preserves the original texture by mapping the $3 \times H \times W$ image colors onto the $XYZ$ coordinates.
* **Interactive Local Dashboard:** Includes a WebGL-powered local dashboard to explore the reconstructed `.glb` scenes, raw point clouds, and voxel meshes directly in the browser.

---

## 💻 Setup & Installation (Windows/PC)

### Prerequisites
* **Python 3.11** (Highly recommended for dependency compatibility)
* **Git**
* *(Optional but recommended)* NVIDIA GPU with CUDA support for faster inference.

### 1. Clone the Repository
```bash
git clone https://github.com/Siddhazntx/Three-D
cd 3D-Reconstruction-Pipeline
```

### 2. Clone Depth-Anything-3 Repository
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V3.git Depth-Anything-3
cd Depth-Anything-3
pip install -e .
cd ..
```

### 3. Create a Virtual Environment
It is best practice to isolate the project dependencies.

```bash
python -m venv minienv
.\minienv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ How to Run

### Generate 3D Models
Place your target 2D image (e.g., `.jpg`, `.png`) into the `data/Sample_img/` directory.

Run the main reconstruction script:

```bash
python da_3d_reconstruction.py
```

The script will process the image, estimate depth, and export the 3D files to the `results/` folder.

### Launch the 3D Web Dashboard
To view the interactive `.glb` and `.ply` files without needing external 3D software:

Ensure you are in the root directory of the project.

Start a local Python web server:

```bash
python -m http.server 8000
```

Open your web browser and navigate to: `http://localhost:8000`

---

## 🛠️ Technical Pipeline
The reconstruction follows a four-step process:

1. **Pre-processing:** Input images are normalized and resized to a $3 \times H \times W$ tensor.
2. **Inference:** The DA3 model predicts a $1 \times H \times W$ depth map representing relative or metric distance.
3. **Geometric Projection:** Using the camera intrinsic matrix $K$, we calculate:
   $$Z = depth$$
   $$X = \frac{(u - c_x) \times Z}{f_x}$$
   $$Y = \frac{(v - c_y) \times Z}{f_y}$$
4. **Post-processing:** Noise filtering and export to the `/result` folder.

---

## 📁 Repository Structure
```text
├── data/               # Input 2D images (RGB)
├── result/             # Generated Depth Maps and .ply Point Clouds
├── src/                # Core model architecture (DA3)
├── da_3d_reconstruction.py  # Main execution script
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
 ```


## 📊 Results

| Input Image (2D) | Reconstructed Point Cloud (3D) |
| :---: | :---: |
| ![Input](data/Sample_img/image1.jpg) | ![Output](results/Sample_img/depth_vis/0000.jpg) |

> **Note:** The output above shows the spatial depth consistency achieved by the Depth Anything V3 backbone. The 3D point cloud was visualized using Open3D.
> > [!IMPORTANT]
> **Full 3D Assets:** Due to the large file size of high-resolution 3D models, the `.glb` and `.ply` files are stored locally. 
> * **Path:** `results/Sample_img`
> * **Format:** Binary GLTF (includes mesh + RGB textures)
