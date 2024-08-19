import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import os

class RealSenseCameraROI:
    def __init__(self, roi=(400, 0, 450, 600, 0.2, 1.2), display_all=False):
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

        # BGR을 RGB로 변환
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # ROI 적용
        x, y, w, h, d_min, d_max = self.roi
        depth_image = depth_image[y:y+h, x:x+w].copy()
        color_image = color_image[y:y+h, x:x+w].copy()

        # 깊이 필터링 적용
        depth_image = np.where((depth_image >= d_min * 1400) & (depth_image <= d_max * 1400), depth_image, 0)

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

        filename = f"saved_pcd/{prefix}_{self.index}.pcd"
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # PCD 파일 저장 시도
        try:
            success = o3d.io.write_point_cloud(filename, pcd)
            if success:
                print(f"Saved {filename} at {position} position")
            else:
                print(f"Failed to save {filename}")
                return False
        except Exception as e:
            print(f"Exception occurred while saving {filename}: {e}")
            return False

        self.index += 1
        return filename

    def draw_roi_box(self, image):
        x, y, w, h, d_min, d_max = self.roi
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return image

    def stop(self):
        self.pipeline.stop()
