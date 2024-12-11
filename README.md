# 3D Reconstruction with RealSense

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

The purpose of this project is to create high-quality 3D models from real-world objects by capturing point cloud data, processing it with filters, and merging it to produce a final mesh. The workflow is optimized for RealSense depth cameras and Open3D for 3D visualization and processing.

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

- Python 3.8 or higher
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
    - Install [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense).

---

## Usage

### Running the Main Pipeline

1. Start the pipeline by running:
    ```bash
    python main.py
    ```
2. Follow the instructions in the console:
    - Press `f` to start capturing frames.
    - Press `ESC` to exit.

### Workflow Steps
1. **Capture**: Captures raw point cloud data (`data/raw`).
2. **Filter**: Processes raw point clouds to remove noise and outliers (`data/filtered`).
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
