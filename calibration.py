import cv2
import numpy as np
import pyrealsense2 as rs
import os

def capture_images_from_both_cameras():
    ctx = rs.context()
    devices = ctx.query_devices()
    assert len(devices) >= 2, "두 대의 카메라가 필요합니다."

    pipeline1 = rs.pipeline()
    pipeline2 = rs.pipeline()
    
    config1 = rs.config()
    config2 = rs.config()
    
    config1.enable_device(devices[0].get_info(rs.camera_info.serial_number))
    config2.enable_device(devices[1].get_info(rs.camera_info.serial_number))
    
    config1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    pipeline1.start(config1)
    pipeline2.start(config2)

    os.makedirs('cameraCapture', exist_ok=True)

    images_cam1 = []
    images_cam2 = []

    try:
        while True:
            frames1 = pipeline1.wait_for_frames()
            frames2 = pipeline2.wait_for_frames()
            
            color_frame1 = frames1.get_color_frame()
            color_frame2 = frames2.get_color_frame()
            
            if not color_frame1 or not color_frame2:
                continue

            image1 = np.asanyarray(color_frame1.get_data())
            image2 = np.asanyarray(color_frame2.get_data())

            # 두 카메라 뷰어를 동시에 표시
            cv2.imshow('Camera 1 Viewer', image1)
            cv2.imshow('Camera 2 Viewer', image2)

            key = cv2.waitKey(1)
            if key == ord('c'):
                filename1 = f'cameraCapture/Camera1_capture_{len(images_cam1)}.png'
                filename2 = f'cameraCapture/Camera2_capture_{len(images_cam2)}.png'
                cv2.imwrite(filename1, image1)
                cv2.imwrite(filename2, image2)
                images_cam1.append(image1)
                images_cam2.append(image2)
                print(f"Captured and saved: {filename1}, {filename2}")
            elif key == ord('q'):
                break
            elif key == ord('f'):
                print("Starting calibration with captured images...")
                calibrate_with_saved_images()
                break
    finally:
        pipeline1.stop()
        pipeline2.stop()
        cv2.destroyAllWindows()

    return images_cam1, images_cam2

def calibrate_with_saved_images():
    pattern_size = (11, 8)
    square_size = 0.015  # 25mm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3D points in real world space
    imgpoints_cam1 = []  # 2D points in image plane cam1
    imgpoints_cam2 = []  # 2D points in image plane cam2

    # Camera 1 and Camera 2 images
    num_images = 14  # or however many images you captured
    for i in range(num_images):
        img1 = cv2.imread(f'cameraCapture/Camera1_capture_{i}.png')
        img2 = cv2.imread(f'cameraCapture/Camera2_capture_{i}.png')
        
        if img1 is None or img2 is None:
            print(f"Image pair {i} not found or could not be loaded.")
            continue

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size, None)

        if ret1 and ret2:
            objpoints.append(objp)
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            imgpoints_cam1.append(corners1)
            imgpoints_cam2.append(corners2)
        else:
            print(f"Chessboard corners not found in image pair {i}.")
    
    if len(objpoints) == 0 or len(imgpoints_cam1) == 0 or len(imgpoints_cam2) == 0:
        print("No valid image pairs found for calibration.")
        return

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)
    
    mtx1 = np.array([[644.0332069319121, 0, 638.01599225990026],
                     [0, 644.08112976982113, 346.6885649322208],
                     [0, 0, 1]])
    dist1 = np.array([-0.030344228094688598, 0.024205395224309, -0.0014208673049117034, -0.0028596856032881658, 0])

    mtx2 = np.array([[649.99621575792867, 0, 649.23332445990832],
                     [0, 649.88947887728057, 358.06202375035213],
                     [0, 0, 1]])
    dist2 = np.array([-0.025687671705320849, 0.023032859726674786, -0.0058890471355568293, -0.002327228599279444, 0])

    ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        objpoints, imgpoints_cam1, imgpoints_cam2,
        mtx1, dist1, mtx2, dist2, gray1.shape[::-1],
        criteria_stereo, flags
    )

    print("Calibration complete.")
    print("Rotation matrix:\n", R)
    print("Translation vector:\n", T)

    np.savez('external_calibration.npz', R=R, T=T, mtx1=mtx1, dist1=dist1, mtx2=mtx2, dist2=dist2)


def main():
    print("Press 'c' to capture images, 'f' to start calibration, or 'q' to quit:")
    capture_images_from_both_cameras()

if __name__ == "__main__":
    main()
