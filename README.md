# 3D Reconstruction with RealSense
## Using of 3D Library Open3D

This project focuses on capturing, filtering, merging, and processing point clouds to generate 3D reconstructions of objects or scenes using RealSense cameras.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Examples](#examples)
7. [Future Work](#future-work)
8. [License](#license)

---

## Introduction

The purpose of this project is to create high-quality 3D models from real-world objects by capturing point cloud data, processing it with filters, and merging it to produce a final mesh.
The workflow is optimized for RealSense depth cameras and Open3D for 3D visualization and processing.
Im use RealSense D455 depth camera

---

## Features

- **Real-time point cloud capture**: Captures depth and color data from RealSense cameras.
- **ROI-based filtering**: Focuses on specific regions for point cloud generation.
- **Outlier removal**: Removes noise and unwanted points from the data.
- **Point cloud merging**: Combines multiple frames into a single aligned point cloud.
- **Mesh generation**: Produces 3D meshes from processed point clouds.
- **Visualization**: Displays intermediate and final results.

---

## Installation

### Prerequisites

- Python 3.8 - 3.10
- Intel RealSense SDK
- Open3D
- NumPy
- OpenCV

### Installation Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/3DReconstruction.git
    cd 3DReconstruction
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your RealSense environment:
    - Ensure your RealSense camera is connected.
    - It needs over USB Port 3.0
    - Install [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense).

---

## Usage

### Running the Main Pipeline

1. Start the pipeline by running:
    ```bash
    python capture_main.py
    ```
2. Follow the instructions in the console:
    - Press `f` to start capturing frames.
    - Press `ESC` to exit.
  
3. You can see CameraViewer through opencv viewer with ROI

### Workflow Steps
1. **Capture**: Captures raw point cloud data (`data/raw`).
2. **Filter**: Processes raw point clouds to remove noise and outliers[Remove background/ground/color Point] (`data/filtered`).
3. **Merge**: Merges filtered frames into a single aligned point cloud (`data/merge_output`).
4. **Mesh**: Generates a 3D mesh from the merged point cloud.

---

## Project Structure

```plaintext
3DReconstruction/
│
├── data/
│   ├── raw/                # Raw point cloud data
│   ├── filtered/           # Filtered point clouds
│   ├── merge_output/       # Merged and cleaned point cloud files
│   └── output/             # Final 3D mesh files
│
├── src/
│   ├── capture.py          # Captures raw point clouds
│   ├── filter.py           # Filters raw point clouds
│   ├── merge.py            # Merges multiple point clouds
│   ├── mesh.py             # Generates meshes from merged clouds
│   └── utils.py            # Utility functions
│
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── main.py                 # Main script to run the pipeline

---

# Method Details

## 1. `RealSenseCameraROI`

### Description:
Captures point cloud data from a RealSense camera[D455, D435 ...], with support for cropping to a specified Region of Interest (ROI).
Using ROI, it can remove unnecessary 3D Point Data. So we save CPU resources.

### Parameters:
- `transform_matrix`: Transformation matrix to align the point cloud data. This Matrix is ​​an external matrix for the camera and world coordinate system and serves to unify the point cloud and camera coordinate system.
- `roi`: Tuple specifying the cropping box `(x, y, width, height)`.

### How It Works:
1. Connects to a RealSense camera and starts streaming depth and color data.
2. Aligns depth data to the color frame for proper mapping.
3. Crops the image and depth data to the specified ROI.
4. Optionally applies a transformation to the resulting point cloud.

### Example Usage:
```python
# Initialize the RealSenseCameraROI class
Example
camera = RealSenseCameraROI(
    transform_matrix=[
        [-0.0000, -0.0000,  1.0000,  -0.94],
        [ 0.0000, -1.0000, -0.0000,   0.100],
        [ 1.0000,  0.0000,  0.0000,   1.53],
        [ 0.0000,  0.0000,  0.0000,   1.0000],
    ],
    roi=(220, 40, 720, 580)
)

# Capture 34 frames
camera.capture_frames(frame_count=34)
```

---

## 2. `filter_pcd_files`

### Description:
Filters point clouds by removing noise, background, and outliers.

### Parameters:
- `input_folder`: Directory containing raw point cloud files.
- `output_folder`: Directory where filtered point clouds will be saved.
- `lower_green`: Lower bound for green color in HSV (used for background removal).
- `upper_green`: Upper bound for green color in HSV.
- `z_threshold`: Threshold to remove points below a certain z-axis value.
- `nb_neighbors`: Number of neighbors for statistical outlier removal.
- `std_ratio`: Standard deviation ratio for statistical outlier removal.

### How It Works:
1. Loads raw point cloud files.
2. Removes points based on color filtering (e.g., green backgrounds).
3. Removes points below a specified z-axis threshold.
4. Applies statistical outlier removal to clean noise.

### Example Usage:
```python
from src.filter import filter_pcd_files

# Filter raw point cloud files
filter_pcd_files(
    input_folder="data/raw",
    output_folder="data/filtered",
    lower_green=(30, 30, 30),
    upper_green=(90, 255, 255),
    z_threshold=0,
    nb_neighbors=50,
    std_ratio=2.0
)
```

---

## 3. `merge_pcd_files`

### Description:
Merges multiple filtered point clouds into a single aligned point cloud.

### Parameters:
- `directory`: Directory containing filtered point cloud files.
- `nb_neighbors`: Number of neighbors for statistical outlier removal.
- `std_ratio`: Standard deviation ratio for statistical outlier removal.

### How It Works:
1. Loads filtered point cloud files.
2. Aligns and merges point clouds
3. Align the point cloud by rotating it according to the turntable speed.
4. Outputs a single merged point cloud.

### Example Usage:
```python
from src.merge import merge_pcd_files

# Merge filtered point clouds
merged_pcd = merge_pcd_files(directory="data/filtered")
```

---

## 4. `final_cleanup`

### Description:
Removes residual outliers and saves the cleaned, merged point cloud.

### Parameters:
- `input_pcd`: Merged point cloud file to be cleaned.

### How It Works:
1. Applies statistical outlier removal to the merged point cloud.
2. Saves the cleaned point cloud to `data/merge_output`.

### Example Usage:
```python
from src.merge import final_cleanup

# Perform final cleanup on the merged point cloud
final_cleanup(input_pcd="data/merge_output/final_merged_model.pcd")
```

---

## 5. `create_mesh`

### Description:
Generates a 3D mesh from the merged and cleaned point cloud.

### Parameters:
- `input_pcd`: Cleaned point cloud file.
- `output_mesh`: File path to save the resulting 3D mesh.
- `depth`: Poisson reconstruction depth.
- `scale`: Poisson reconstruction scale.

### How It Works:
1. Loads the cleaned point cloud.
2. Applies Poisson surface reconstruction to generate the mesh.
3. Saves the mesh file.

### Example Usage:
```python
from src.mesh import create_mesh

# Generate a 3D mesh from the cleaned point cloud
create_mesh(
    input_pcd="data/merge_output/final_cleaned_model.pcd",
    output_mesh="data/output/final_mesh.ply",
    depth=10,
    scale=1.5
)
```

---

## Notes
- Ensure the RealSense camera is connected before running capture functions.
- Check the `data` directory for intermediate and final results after running the pipeline.
- Customize the parameters in each method to fit your specific use case.


