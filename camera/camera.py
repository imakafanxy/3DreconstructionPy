import time
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import os

def start_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    return pipeline, profile

def stop_camera(pipeline):
    pipeline.stop()

def capture_images(pipeline, num_images=8, interval=5, save_dir="saved_pcd"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(num_images):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
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
        
        o3d.io.write_point_cloud(f"{save_dir}/outputCloud_{i}.pcd", pcd)
        print(f"Saved: {save_dir}/outputCloud_{i}.pcd")
        time.sleep(interval)

if __name__ == "__main__":
    pipeline, profile = start_camera()
    try:
        capture_images(pipeline)
    finally:
        stop_camera(pipeline)
