import open3d as o3d
import numpy as np
import os
from sklearn.cluster import DBSCAN

def load_point_clouds(directory, filenames=None, voxel_size=0.05):
    pcds = []
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        print(f"Loading file: {filepath}")
        pcd = o3d.io.read_point_cloud(filepath)
        if not pcd.is_empty():
            print(f"Point cloud loaded with {len(pcd.points)} points.")
            pcd = pcd.voxel_down_sample(voxel_size)
            pcds.append(pcd)
        else:
            print(f"Warning: {filename} is empty or could not be loaded.")
    return pcds

def remove_background(pcd, distance_threshold=1.2):
    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points, axis=1)
    mask = distances < distance_threshold
    pcd = pcd.select_by_index(np.where(mask)[0])
    return pcd

def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    if not source.has_normals():
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max_correspondence_distance_coarse * 2, max_nn=30))
    if not target.has_normals():
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max_correspondence_distance_coarse * 2, max_nn=30))

    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine, icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    for source_id in range(len(pcds)):
        for target_id in range(source_id + 1, len(pcds)):
            transformation_icp, information_icp = pairwise_registration(pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
            if target_id == source_id + 1:
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, transformation_icp, information_icp, uncertain=False))
            else:
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, transformation_icp, information_icp, uncertain=True))
    return pose_graph

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = pcd.select_by_index(ind)
    return pcd

def remove_small_clusters(pcd, eps=0.05, min_samples=10):
    points = np.asarray(pcd.points)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]
    largest_cluster_mask = labels == largest_cluster_label
    pcd = pcd.select_by_index(np.where(largest_cluster_mask)[0])
    return pcd

def remove_density_outliers(pcd, radius=0.05, min_neighbors=5):
    _, ind = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    return pcd.select_by_index(ind)

def remove_normal_outliers(pcd, threshold=0.8):
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)
    mean_normal = np.mean(normals, axis=0)
    deviations = np.linalg.norm(normals - mean_normal, axis=1)
    mask = deviations < threshold
    return pcd.select_by_index(np.where(mask)[0])

def save_combined_point_cloud(pcd_combined, filename):
    if not pcd_combined.is_empty():
        success = o3d.io.write_point_cloud(filename, pcd_combined)
        if success:
            print(f"Aligned point cloud saved successfully to {filename}")
        else:
            print(f"Failed to save aligned point cloud to {filename}")
    else:
        print("Combined point cloud is empty. Nothing to save.")
