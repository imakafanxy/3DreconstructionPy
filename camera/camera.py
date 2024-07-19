import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import os

class RealSenseCamera:
    def __init__(self):
        """ 초기화 함수: RealSense 카메라 파이프라인 및 설정 초기화 """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.index = 1
        if not os.path.exists("weseePj/saved_pcd"):
            os.makedirs("weseePj/saved_pcd")

    def capture_frame(self):
        """ 실시간 프레임 캡처 및 저장: depth와 color 프레임을 캡처하고 .pcd 파일로 저장 """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Filter out points beyond 1 meter
        depth_image[depth_image > 1000] = 0

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

if __name__ == "__main__":
    camera = RealSenseCamera()
    try:
        while True:
            frames = camera.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Filter out points beyond 1 meter for display
            depth_display = np.copy(depth_image)
            depth_display[depth_display > 1000] = 0

            # Normalize depth image for display
            depth_display = cv2.normalize(depth_display, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = np.uint8(depth_display)
            depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            combined_image = np.hstack((color_image, depth_display))

            cv2.imshow('RealSense', combined_image)

            key = cv2.waitKey(1)
            if key == ord('s'):
                if camera.capture_frame():
                    print("Captured frame")
                else:
                    print("Failed to capture frame")
            elif key == 27:  # ESC 키
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
