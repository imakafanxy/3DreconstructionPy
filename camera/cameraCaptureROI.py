import open3d as o3d
import numpy as np
import cv2
import os

class RealSenseCameraROI:
    def __init__(self):
        # RealSenseSensor 초기화 및 설정
        self.sensor = o3d.t.io.RealSenseSensor()
        self.cfg = o3d.t.io.RealSenseSensorConfig()
        # self.cfg.depth_format = o3d.t.io.RealSenseSensorConfig.DepthFormat.Z16
        # self.cfg.color_format = o3d.t.io.RealSenseSensorConfig.ColorFormat.RGB8
        self.cfg.depth_resolution = [640, 480]  # 깊이 해상도 설정
        self.cfg.color_resolution = [640, 480]  # 컬러 해상도 설정
        self.cfg.fps = 30
        self.sensor.start_capture(self.cfg)  # 카메라 스트리밍 시작

        self.index = 1
        self.roi = (200, 100, 650, 400, 0.5, 2.0)  # ROI 초기 설정
        self.pcd_dir = "saved_pcd"
        if not os.path.exists(self.pcd_dir):
            os.makedirs(self.pcd_dir)

    def stream_with_roi(self):
        """ 실시간 스트리밍과 ROI 박스를 표시하는 함수 """
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window('RealSense Stream with ROI')

        try:
            while True:
                # RealSense 프레임 캡처
                rgbd_frame = self.sensor.capture_frame(True, True)
                if rgbd_frame is None:
                    continue

                # 컬러 이미지와 깊이 이미지를 numpy 배열로 변환
                color_image = np.asarray(rgbd_frame.color)
                depth_image = np.asarray(rgbd_frame.depth)

                # ROI 적용
                x, y, w, h, d_min, d_max = self.roi
                depth_image_roi = depth_image[y:y+h, x:x+w]
                color_image_roi = color_image[y:y+h, x:x+w]

                # 깊이 필터 적용 (설정된 깊이 범위 내의 값만 유지)
                depth_filtered = np.where((depth_image_roi >= d_min * 2000) & (depth_image_roi <= d_max * 2000), depth_image_roi, 0)

                # ROI 박스 그리기
                color_image_with_roi = color_image.copy()
                cv2.rectangle(color_image_with_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 컬러 이미지와 깊이 이미지를 나란히 보여줌
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                combined_image = np.hstack((color_image_with_roi, depth_colormap))

                # 실시간 이미지 출력
                cv2.imshow('RealSense Stream with ROI', combined_image)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):  # 'q' 키로 종료
                    break

                # ROI 박스를 조절할 키 입력 처리 (키보드 방향키)
                elif key == ord('w'):  # 위로 이동
                    self.roi = (x, max(0, y - 5), w, h, d_min, d_max)
                elif key == ord('s'):  # 아래로 이동
                    self.roi = (x, min(depth_image.shape[0] - h, y + 5), w, h, d_min, d_max)
                elif key == ord('a'):  # 좌로 이동
                    self.roi = (max(0, x - 5), y, w, h, d_min, d_max)
                elif key == ord('d'):  # 우로 이동
                    self.roi = (min(depth_image.shape[1] - w, x + 5), y, w, h, d_min, d_max)

        finally:
            vis.destroy_window()
            self.sensor.stop_capture()
            cv2.destroyAllWindows()

    def capture_frame(self, prefix="output"):
        """ 프레임을 캡처하고 ROI 내 포인트 클라우드를 저장하는 함수 """
        rgbd_frame = self.sensor.capture_frame(True, True)
        if rgbd_frame is None:
            return False

        # ROI 적용
        x, y, w, h, d_min, d_max = self.roi
        color_image = np.asarray(rgbd_frame.color)[y:y+h, x:x+w]
        depth_image = np.asarray(rgbd_frame.depth)[y:y+h, x:x+w]

        # 깊이 필터 적용
        depth_image = np.where((depth_image >= d_min * 2000) & (depth_image <= d_max * 2000), depth_image, 0)

        # Open3D RGBD 이미지 생성
        rgbd_image = o3d.t.geometry.RGBDImage(o3d.t.geometry.Image(color_image),
                                               o3d.t.geometry.Image(depth_image))

        # 포인트 클라우드 생성
        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        )

        # 포인트 클라우드 저장
        filename = f"{self.pcd_dir}/{prefix}_{self.index}.pcd"
        o3d.t.io.write_point_cloud(filename, pcd)
        self.index += 1
        print(f"Captured and saved: {filename}")
        return filename

    def stop(self):
        """ 스트리밍 종료 함수 """
        self.sensor.stop_capture()
        cv2.destroyAllWindows()
