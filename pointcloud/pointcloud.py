import open3d as o3d
import numpy as np
import os

def load_point_clouds(directory, filenames=None, voxel_size=0.01):
    pcds = []
    for filename in filenames:
        filepath = filename
        print(f"Loading file: {filepath}")
        pcd = o3d.io.read_point_cloud(filepath)
        if not pcd.is_empty():
            pcd = pcd.voxel_down_sample(voxel_size)
            pcds.append(pcd)
        else:
            print(f"Warning: {filename} is empty or could not be loaded.")
    return pcds

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, init_transformation, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return result

def colored_icp_registration(source, target, max_correspondence_distance=0.02, voxel_size=0.01):
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    init_transformation = np.identity(4)
    
    result = o3d.pipelines.registration.registration_colored_icp(
        source_down, target_down, max_correspondence_distance,
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP())
    
    return result.transformation

def sequential_registration(pcds, voxel_size=0.01):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    for i in range(len(pcds) - 1):
        source_down, source_fpfh = preprocess_point_cloud(pcds[i], voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(pcds[i + 1], voxel_size)
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        result_icp = refine_registration(pcds[i], pcds[i + 1], result_ransac.transformation, voxel_size)
        transformation_icp = colored_icp_registration(pcds[i], pcds[i + 1], voxel_size=voxel_size)
        odometry = np.dot(transformation_icp, odometry)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, i + 1, transformation_icp, np.identity(6), uncertain=True))
        print(f"Completed registration for frame pair {i} and {i+1}")
    return pose_graph

def optimize_pose_graph(pose_graph):
    option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=0.02, edge_prune_threshold=0.25, reference_node=0)
    o3d.pipelines.registration.global_optimization(pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)
    return pose_graph

def apply_pose_graph(pcds, pose_graph):
    for point_id in range(len(pose_graph.nodes)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    pcd_combined = o3d.geometry.PointCloud()
    for pcd in pcds:
        pcd_combined += pcd
    return pcd_combined

def save_aligned_cloud(aligned_cloud, filename="weSeePJ/saved_pcd/aligned_output.pcd"):
    if aligned_cloud.is_empty():
        print("Warning: Aligned cloud is empty, not saving.")
        return
    o3d.io.write_point_cloud(filename, aligned_cloud)
    print(f"Aligned point cloud saved as {filename}")

def remove_background(pcd, distance_threshold=1.0):
    distances = np.linalg.norm(np.asarray(pcd.points), axis=1)
    mask = distances < distance_threshold
    pcd_filtered = pcd.select_by_index(np.where(mask)[0])
    return pcd_filtered
