import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import os
import time

class RealSenseCameraSeg:
    def __init__(self):
        """ 초기화 함수: RealSense 카메라 파이프라인 및 설정 초기화 """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.index = 1
        self.net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
        if not os.path.exists("weSeePJ/saved_pcd"):
            os.makedirs("weSeePJ/saved_pcd")

    def capture_frames(self, delay=10, duration=40, interval=1):
        """ 일정 시간 동안 주기적으로 프레임을 캡처 """
        start_time = time.time()
        captured_files = []
        while time.time() - start_time < delay + duration:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 얼굴 검출
            blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()

            # 검출된 얼굴의 바운딩 박스 적용
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([color_image.shape[1], color_image.shape[0], color_image.shape[1], color_image.shape[0]])
                    (x, y, x1, y1) = box.astype("int")
                    color_image = color_image[y:y1, x:x1]
                    depth_image = depth_image[y:y1, x:x1]
                    break

            # 거리 필터링
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense', images)

            if time.time() - start_time > delay:
                filename = self.capture_frame_with_segmentation(depth_image, color_image)
                if filename:
                    captured_files.append(filename)
                time.sleep(interval)

            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break

        return captured_files

    def capture_frame_with_segmentation(self, depth_image, color_image):
        """ Segmentation을 사용하여 프레임 캡처 및 저장 """
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
        pcd = self.filter_by_distance(pcd, max_distance=1.0)

        # 더 정확한 배경 제거
        pcd = self.remove_background(pcd)

        filename = f"weSeePJ/saved_pcd/outputCloud_{self.index}.pcd"
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved {filename}")
        self.index += 1
        return filename

    def filter_by_distance(self, pcd, max_distance=1.0):
        """주어진 거리 이내의 포인트 필터링"""
        distances = np.asarray(pcd.points)[:, 2]
        mask = distances < max_distance
        pcd_filtered = pcd.select_by_index(np.where(mask)[0])
        return pcd_filtered

    def remove_background(self, pcd, voxel_size=0.01):
        """ 포인트 클라우드에서 배경을 제거 """
        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
        largest_cluster_index = np.argmax(np.bincount(labels[labels >= 0]))
        pcd_filtered = pcd.select_by_index(np.where(labels == largest_cluster_index)[0])
        return pcd_filtered

    def stop(self):
        """ 카메라 파이프라인 종료 """
        self.pipeline.stop()

if __name__ == "__main__":
    camera = RealSenseCameraSeg()
    try:
        while True:
            frames = camera.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 거리 필터링
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense', images)

            key = cv2.waitKey(1)
            if key == ord('s'):
                captured_files = camera.capture_frames(delay=10, duration=40, interval=1)
                break
            elif key == 27:  # ESC key
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
