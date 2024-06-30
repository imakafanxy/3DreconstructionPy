import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import os

class RealSenseCameraROI:
    def __init__(self, roi):
        """ 초기화 함수: RealSense 카메라 파이프라인 및 설정 초기화 """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.index = 1  # 1부터 시작
        self.roi = roi
        if not os.path.exists("weSeePJ/saved_pcd"):
            os.makedirs("weSeePJ/saved_pcd")

    def capture_frame_with_roi(self):
        """ ROI를 적용하여 실시간 프레임 캡처 및 저장 """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Apply ROI
        x, y, w, h = self.roi
        depth_image = depth_image[y:y+h, x:x+w]
        color_image = color_image[y:y+h, x:x+w]

        depth_image = np.ascontiguousarray(depth_image)
        color_image = np.ascontiguousarray(color_image)

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
        
        filename = f"weSeePJ/saved_pcd/outputCloud_{self.index}.pcd"
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved {filename}")
        self.index += 1
        return True

    def stop(self):
        """ 카메라 파이프라인 종료 """
        self.pipeline.stop()
