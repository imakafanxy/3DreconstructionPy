import time
import open3d as o3d
import numpy as np
import cv2
import os
from camera.cameraCaptureROI import RealSenseCameraROI
from pointcloud import load_point_clouds, remove_background, save_combined_point_cloud

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd_down

def extract_fpfh(pcd, voxel_size):
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

def register_point_clouds_with_fpfh(pcds, voxel_size, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    pcd_downs = [preprocess_point_cloud(pcd, voxel_size) for pcd in pcds]
    fpfh_features = [extract_fpfh(pcd_down, voxel_size) for pcd_down in pcd_downs]
    
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    
    for source_id in range(len(pcd_downs)):
        for target_id in range(source_id + 1, len(pcd_downs)):
            icp_coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                pcd_downs[source_id], pcd_downs[target_id], fpfh_features[source_id], fpfh_features[target_id], True,
                max_correspondence_distance_coarse,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance_coarse)
                ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

            icp_fine = o3d.pipelines.registration.registration_icp(
                pcd_downs[source_id], pcd_downs[target_id], max_correspondence_distance_fine, icp_coarse.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            
            transformation_icp = icp_fine.transformation
            information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                pcd_downs[source_id], pcd_downs[target_id], max_correspondence_distance_fine, icp_fine.transformation)
            
            if target_id == source_id + 1:
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, transformation_icp, information_icp, uncertain=False))
            else:
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, transformation_icp, information_icp, uncertain=True))
    
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(pose_graph,
                                                   o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                                   o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                   option)
    
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcd_downs)):
        pcd_downs[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcd_downs[point_id]
    
    return pcd_combined

def main():
    camera = RealSenseCameraROI(display_all=True)
    max_correspondence_distance_coarse = 0.1
    max_correspondence_distance_fine = 0.05
    voxel_size = 0.05
    directory = "saved_pcd"
    
    if not os.path.exists(directory):
        
        os.makedirs(directory)
    
    try:
        captured_files = []
        capturing = False
        capture_start_time = 0
        
        while True:
            frames = camera.pipeline.wait_for_frames()
            aligned_frames = camera.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            color_image = camera.draw_roi_box(color_image)

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            combined_image = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense', combined_image)

            key = cv2.waitKey(1)

            if key == ord('c'):
                print("Starting capture in 10 seconds...")
                time.sleep(10)
                capturing = True
                capture_start_time = time.time()

            if capturing:
                current_time = time.time()
                if current_time - capture_start_time > 1.5:
                    filename = camera.capture_frame(prefix="output")
                    captured_files.append(filename)
                    print(f"Captured file: {filename}")
                    capture_start_time = current_time
                    if len(captured_files) >= 30:
                        capturing = False
                        print("Capture complete.")
            elif key == ord('r'):
                captured_files = [f for f in os.listdir(directory) if f.startswith('output') and f.endswith('.pcd')]
                pcds = load_point_clouds(directory, filenames=captured_files, voxel_size=voxel_size)
                
                for pcd in pcds:
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
                    pcd = remove_background(pcd, distance_threshold=1.0)

                print("Registering point clouds with FPFH features...")
                pcd_combined = register_point_clouds_with_fpfh(pcds, voxel_size, max_correspondence_distance_coarse, max_correspondence_distance_fine)
                
                if not pcd_combined.is_empty():
                    print(f"Combined point cloud has {len(pcd_combined.points)} points. Saving...")
                    save_path = f"saved_pcd/front_aligned_output.pcd"
                    save_combined_point_cloud(pcd_combined, save_path)
                else:
                    print("Combined point cloud is empty. Nothing to save.")
            elif key == 27:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
