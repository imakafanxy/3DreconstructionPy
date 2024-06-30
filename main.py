import cv2
import numpy as np
from camera import RealSenseCameraROI, RealSenseCameraSeg
from pointcloud.pointcloud import load_point_clouds, align_point_clouds_sequentially, save_aligned_cloud
from mesh.mesh import create_mesh_from_pcd, save_mesh
import os

def main(mode="ROI"):
    directory = "weSeePJ/saved_pcd"
    if not os.path.exists(directory):
        os.makedirs(directory)

    if mode == "ROI":
        roi = (100, 100, 440, 280)  # 예제 ROI 설정
        camera = RealSenseCameraROI(roi)
    elif mode == "Segmentation":
        camera = RealSenseCameraSeg()
    else:
        print("Invalid mode selected. Choose either 'ROI' or 'Segmentation'.")
        return

    captured_files = []

    try:
        while True:
            frames = camera.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            cv2.imshow('RealSense', color_image)
            key = cv2.waitKey(1)

            if key == ord('s'):
                if mode == "ROI":
                    success = camera.capture_frame_with_roi()
                elif mode == "Segmentation":
                    success = camera.capture_frame_with_segmentation()

                if success:
                    captured_files.append(f"outputCloud_{camera.index - 1}.pcd")
                    print("Captured frame")
                else:
                    print("Failed to capture frame")

                # Check if 8 frames have been captured
                if len(captured_files) == 8:
                    print("8 frames captured, proceeding with alignment and meshing.")
                    break

            elif key == 27:  # ESC key
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    if len(captured_files) < 2:
        print("Not enough frames captured for alignment.")
        return

    # PCD 파일 로드
    pcds = load_point_clouds(directory=directory, filenames=captured_files)

    # PCD 파일 정합
    aligned_cloud = align_point_clouds_sequentially(pcds)

    # 결과 저장
    aligned_pcd_filename = f"{directory}/aligned_output.pcd"
    save_aligned_cloud(aligned_cloud, filename=aligned_pcd_filename)

    # 메쉬 생성 및 저장
    mesh = create_mesh_from_pcd(aligned_pcd_filename)
    save_mesh(mesh, filename=f"{directory}/mesh_output.ply")

if __name__ == "__main__":
    # mode를 "ROI" 또는 "Segmentation"으로 변경하여 실행
    main(mode="Segmentation")  # "Segmentation"으로 변경하여 Segmentation 모드 실행
