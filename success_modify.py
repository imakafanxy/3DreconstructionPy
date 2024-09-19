import time
import cv2
import os
import numpy as np
import pyrealsense2 as rs
import open3d as o3d

class RealSenseCameraROI:
    def __init__(self, roi=(200, 100, 650, 400, 0.2, 1.2), display_all=False):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(self.config)
        self.index = 1
        self.roi = roi
        self.display_all = display_all
        if not os.path.exists("saved_pcd"):
            os.makedirs("saved_pcd")

    def capture_frame(self, prefix="output", position="front"):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # ROI 적용
        x, y, w, h, d_min, d_max = self.roi
        depth_image = depth_image[y:y+h, x:x+w].copy()
        color_image = color_image[y:y+h, x:x+w].copy()

        # 깊이 필터링 적용
        depth_image = np.where((depth_image >= d_min * 2000) & (depth_image <= d_max * 2000), depth_image, 0)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image),
            convert_rgb_to_intensity=False
            )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            )
        )
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

    def draw_roi_box(self, image):
        x, y, w, h, d_min, d_max = self.roi
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return image

    def stop(self):
        self.pipeline.stop()

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

def remove_small_clusters(pcd, eps=0.05, min_samples=10):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples))
    major_cluster = labels == np.argmax(np.bincount(labels[labels >= 0]))
    return pcd.select_by_index(np.where(major_cluster)[0])

def remove_background(pcd, distance_threshold=1.2):
    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points, axis=1)
    mask = distances < distance_threshold
    pcd = pcd.select_by_index(np.where(mask)[0])
    return pcd

def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max_correspondence_distance_coarse * 2, max_nn=30))
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

def main():
    camera = RealSenseCameraROI(display_all=True)
    max_correspondence_distance_coarse = 0.05
    max_correspondence_distance_fine = 0.02
    voxel_size = 0.05
    directory = "saved_pcd"
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        captured_files = []

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
            
            if key == ord('f'):
                # 10초 대기
                print("Waiting for 10 seconds before starting capture...")
                time.sleep(10)

                # 1초에 2장씩, 40초 동안 촬영 (총 80장)
                start_time = time.time()
                capturing = True
                frame_count = 0

                while capturing:
                    if time.time() - start_time > 40:
                        print("Capture complete.")
                        capturing = False
                        break

                
                    filename = camera.capture_frame(prefix="output", position="front")

                    captured_files.append(filename)
                    print(f"Captured file {frame_count+1}: {filename}")
                    
                    # 0.5초 대기 (1초에 2장 촬영을 위해)
                    time.sleep(0.5)
                    frame_count += 1
                
            elif key == ord('q'):
                # 로드한 PCD 전처리 및 정합
                print(f"Processing files: {captured_files}")
                captured_files = [f for f in os.listdir(directory) if f.startswith('output') and f.endswith('.pcd')]
                pcds = [o3d.io.read_point_cloud(os.path.join(directory, f)) for f in captured_files]

                # 전처리
                # for pcd in pcds:
                #     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
                #     pcd = remove_background(pcd, distance_threshold=1.2)
                #     pcd = remove_outliers(pcd)
                #     pcd = remove_small_clusters(pcd)

                # 정합
                pose_graph = full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine)
                option = o3d.pipelines.registration.GlobalOptimizationOption(
                    max_correspondence_distance=max_correspondence_distance_fine,
                    edge_prune_threshold=0.25,
                    reference_node=0)
                o3d.pipelines.registration.global_optimization(pose_graph,
                                                                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                                                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                                option)
                pcd_combined = o3d.geometry.PointCloud()
                for point_id in range(len(pcds)):
                    pcds[point_id].transform(pose_graph.nodes[point_id].pose)
                    pcd_combined += pcds[point_id]

                if not pcd_combined.is_empty():
                    print(f"Combined point cloud has {len(pcd_combined.points)} points. Saving...")
                    save_path = f"saved_pcd/aligned_output.pcd"
                    success = o3d.io.write_point_cloud(save_path, pcd_combined)
                    if success:
                        print(f"Aligned point cloud saved successfully to {save_path}")
                    else:
                        print(f"Failed to save aligned point cloud to {save_path}")
                else:
                    print("Combined point cloud is empty. Nothing to save.")

            elif key == 27:  # ESC to exit
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
