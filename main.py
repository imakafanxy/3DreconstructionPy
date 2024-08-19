import time
import cv2
import numpy as np
import os
import open3d as o3d
from camera.cameraCaptureROI import RealSenseCameraROI
from pointcloud import load_point_clouds, remove_background, full_registration, save_combined_point_cloud, remove_outliers, remove_small_clusters, remove_density_outliers, remove_normal_outliers

def main():
    camera = RealSenseCameraROI(display_all=True)
    max_correspondence_distance_coarse = 0.03
    max_correspondence_distance_fine = 0.01
    voxel_size = 0.05
    directory = "saved_pcd"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    try:
        front_captured_files = []
        back_captured_files = []
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
                    if len(captured_files) >= 25:
                        capturing = False
                        print("Capture complete.")
                        
            if key == ord('r'):
                print(f"Processing front files: {front_captured_files}")
                front_captured_files = [f for f in os.listdir(directory) if f.startswith('output') and f.endswith('.pcd')]
                pcds = load_point_clouds(directory, filenames=front_captured_files, voxel_size=voxel_size)
                
                for pcd in pcds:
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
                    pcd = remove_background(pcd, distance_threshold=1.0)

                print(1)
                pose_graph = full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine)
                option = o3d.pipelines.registration.GlobalOptimizationOption(
                    max_correspondence_distance=max_correspondence_distance_fine,
                    edge_prune_threshold=0.25,
                    reference_node=0)
                print(2)
                o3d.pipelines.registration.global_optimization(pose_graph,
                                                                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                                                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                                option)
                print(3)
                pcd_combined = o3d.geometry.PointCloud()
                for point_id in range(len(pcds)):
                    pcds[point_id].transform(pose_graph.nodes[point_id].pose)
                    pcd_combined += pcds[point_id]
                print(4)
                
                #이상치 제거 및 작은 클러스터 제거
                pcd_combined = remove_outliers(pcd_combined)
                print(f"Combined point cloud has {len(pcd_combined.points)} points after outlier removal.")
                
                pcd_combined = remove_small_clusters(pcd_combined)
                print(f"Combined point cloud has {len(pcd_combined.points)} points after small cluster removal.")
                
                pcd_combined = remove_density_outliers(pcd_combined, radius=0.05, min_neighbors=5)
                print(f"Combined point cloud has {len(pcd_combined.points)} points after density outlier removal.")
                
                #pcd_combined = remove_normal_outliers(pcd_combined, threshold=0.3)
                print(f"Combined point cloud has {len(pcd_combined.points)} points after normal outlier removal.")

                if not pcd_combined.is_empty():
                    print(f"Combined point cloud has {len(pcd_combined.points)} points. Saving...")
                    save_path = f"saved_pcd/front_aligned_output.pcd"
                    success = o3d.io.write_point_cloud(save_path, pcd_combined)
                    if success:
                        print(f"Aligned point cloud saved successfully to {save_path}")
                    else:
                        print(f"Failed to save aligned point cloud to {save_path}")
                else:
                    print("Combined point cloud is empty. Nothing to save.")

            elif key == 27:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
