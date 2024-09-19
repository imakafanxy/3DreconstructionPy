import time
import cv2
import numpy as np
import os
import open3d as o3d
from camera.cameraCaptureROI import RealSenseCameraROI
from pointcloud import load_and_preprocess_point_clouds, full_registration

def main():
    camera = RealSenseCameraROI()
    voxel_size = 0.05  
    max_correspondence_distance_coarse = 0.01  
    max_correspondence_distance_fine = 0.05  

    directory = "saved_pcd"
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    capturing = False
    frame_count = 0
    captured_files = []

    try:
        while True:
            # Stream 이미지와 ROI 박스 표시
            camera.stream_with_roi()

            key = cv2.waitKey(1)

            if key == ord('c'):
                print("Waiting for 10 seconds before starting capture...")
                time.sleep(10)

                start_time = time.time()
                capturing = True
                frame_count = 0

                while capturing:
                    if time.time() - start_time > 40:
                        print("Capture complete.")
                        capturing = False
                        break

                    filename = camera.capture_frame(prefix=f"output_{frame_count + 1}")

                    if not filename:
                        print(f"[ERROR] Capture failed for frame {frame_count + 1}.")
                    else:
                        captured_files.append(filename)
                        print(f"Captured file {frame_count + 1}: {filename}")
                        frame_count += 1

                    time.sleep(0.5)

            elif key == ord('r'):
                print("Starting registration...")

                pcd_files = os.listdir(directory)
                pcd_files.sort()

                if len(pcd_files) == 0:
                    print("No processed point clouds found. Please capture first.")
                    continue

                # load_and_preprocess_point_clouds 함수를 호출하여 전처리 수행
                pcds = load_and_preprocess_point_clouds(directory, filenames=pcd_files, voxel_size=voxel_size)
                total_files = len(pcds)
                print(f"Loaded {total_files} point clouds for registration.")

                start_time = time.time()
                print("Performing global registration...")

                # full_registration 함수 호출
                pcd_combined, transformations = full_registration(pcds, voxel_size, max_correspondence_distance_fine)
                
                registration_time = time.time() - start_time
                print(f"Global registration and fine ICP completed in {registration_time:.2f} seconds.")

                if not pcd_combined.is_empty():
                    print(f"Final merged point cloud contains {len(pcd_combined.points)} points.")

                    print("Starting to save final PCD...")
                    start_save_time = time.time()

                    final_save_path = os.path.join(directory, "final_output.pcd")
                    o3d.io.write_point_cloud(final_save_path, pcd_combined)

                    save_time = time.time() - start_save_time
                    print(f"Final aligned point cloud saved to {final_save_path} in {save_time:.2f} seconds.")
                else:
                    print("[ERROR] Merged point cloud is empty, skipping save.")

            elif key == ord('q'):
                print("Exiting.")
                break

    finally:
        camera.sensor.stop_capture()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
