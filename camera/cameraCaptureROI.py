import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import os
import time

class RealSenseCameraROI:
    def __init__(self, roi=(50, 50, 300, 400, 0.2, 1.0)):  # 원하는 ROI 값으로 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(self.config)
        self.index = 1
        self.roi = roi
        if not os.path.exists("weSeePJ/saved_pcd"):
            os.makedirs("weSeePJ/saved_pcd")

    def capture_frames(self, delay=10, duration=40, interval=1):
        time.sleep(delay)  # 촬영 준비 시간
        start_time = time.time()
        captured_files = []
        while time.time() - start_time < duration:
            filename = self.capture_frame_with_roi()
            if filename:
                captured_files.append(filename)
            time.sleep(interval)
        return captured_files

    def capture_frame_with_roi(self):
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
        depth_image = np.where((depth_image >= d_min * 1000) & (depth_image <= d_max * 1000), depth_image, 0)

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
        
        # 거리 필터링
        pcd = self.filter_by_distance(pcd, min_distance=d_min, max_distance=d_max)
        
        filename = f"weSeePJ/saved_pcd/outputCloud_{self.index}.pcd"
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved {filename}")
        self.index += 1
        return filename

    def filter_by_distance(self, pcd, min_distance, max_distance):
        points = np.asarray(pcd.points)
        distances = np.linalg.norm(points, axis=1)
        mask = (distances >= min_distance) & (distances <= max_distance)
        pcd_filtered = pcd.select_by_index(np.where(mask)[0])
        return pcd_filtered

    def stop(self):
        self.pipeline.stop()

if __name__ == "__main__":
    camera = RealSenseCameraROI()
    try:
        while True:
            frames = camera.pipeline.wait_for_frames()
            aligned_frames = camera.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # BGR을 RGB로 변환
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # ROI 적용
            x, y, w, h, d_min, d_max = camera.roi
            roi_color_image = color_image[y:y+h, x:x+w]
            roi_depth_image = depth_image[y:y+h, x:x+w]

            # ROI 영역 시각적으로 표시
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Filter out points beyond 1 meter for display
            depth_display = np.copy(roi_depth_image)
            depth_display[depth_display > 1000] = 0

            # Normalize depth image for display
            depth_display = cv2.normalize(depth_display, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = np.uint8(depth_display)
            depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            # Adjust image size for square display
            height, width, _ = color_image.shape
            size = min(height, width)
            roi_color_image = cv2.resize(roi_color_image, (size, size))
            depth_display = cv2.resize(depth_display, (size, size))

            # Combine images side by side
            combined_image = np.hstack((roi_color_image, depth_display))

            # Create a larger window with appropriate size
            cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('RealSense', 1280, 720)
            cv2.imshow('RealSense', combined_image)

            key = cv2.waitKey(1)
            if key == ord('s'):
                camera.capture_frames(delay=10, duration=40, interval=1)
                break
            elif key == 27:  # ESC key
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
