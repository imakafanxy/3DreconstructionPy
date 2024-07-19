import cv2
from camera.cameraCaptureSeg import RealSenseCameraSeg
from camera.cameraCaptureROI import RealSenseCameraROI
from pointcloud.pointcloud import load_point_clouds, sequential_registration, optimize_pose_graph, apply_pose_graph, save_aligned_cloud, remove_background
from mesh.mesh import create_mesh_from_pcd, save_mesh
import open3d as o3d
import numpy as np
import os

def denoise_and_filter_pcd(pcd, distance_threshold=1.0, nb_neighbors=20, std_ratio=2.0):
    pcd = remove_background(pcd, distance_threshold)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = pcd.select_by_index(ind)
    return pcd

def main(mode="ROI"):
    directory = "weSeePJ/saved_pcd"
    if not os.path.exists(directory):
        os.makedirs(directory)

    if mode == "Segmentation":
        camera = RealSenseCameraSeg()
    elif mode == "ROI":
        camera = RealSenseCameraROI()
    else:
        print("Invalid mode selected. Choose 'Segmentation' or 'ROI'.")
        return

    captured_files = camera.capture_frames(delay=10, duration=40, interval=1)
    print(f"Captured files: {captured_files}")

    if len(captured_files) < 2:
        print("Not enough frames captured for alignment.")
        return

    pcds = load_point_clouds(directory, filenames=captured_files)

    pose_graph = sequential_registration(pcds)
    print("Initial pose graph created with {} nodes and {} edges.".format(len(pose_graph.nodes), len(pose_graph.edges)))

    optimized_pose_graph = optimize_pose_graph(pose_graph)
    print("Pose graph optimized.")

    aligned_cloud = apply_pose_graph(pcds, optimized_pose_graph)
    print("Pose graph applied and point clouds aligned.")

    print("Denoising and filtering point cloud...")
    aligned_cloud = denoise_and_filter_pcd(aligned_cloud, distance_threshold=1.0)

    aligned_pcd_filename = os.path.join(directory, "aligned_output.pcd")
    print(f"Saving aligned cloud to {aligned_pcd_filename}...")
    save_aligned_cloud(aligned_cloud, filename=aligned_pcd_filename)

    print("Creating mesh from aligned point cloud...")
    mesh = create_mesh_from_pcd(aligned_pcd_filename)
    if mesh is None:
        print("Failed to create mesh.")
        return
    mesh_filename = os.path.join(directory, "mesh_output.ply")
    print(f"Saving mesh to {mesh_filename}...")
    save_mesh(mesh, filename=mesh_filename)
    print("Mesh created and saved.")

if __name__ == "__main__":
    main(mode="ROI")
