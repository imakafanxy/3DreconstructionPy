import open3d as o3d
import numpy as np

def load_point_clouds(directory="weSeePJ/saved_pcd", filenames=None, voxel_size=0.02):
    """ 포인트 클라우드 파일 로드 및 다운샘플링 """
    pcds = []
    for filename in filenames:
        pcd = o3d.io.read_point_cloud(f"{directory}/{filename}")
        if not pcd.is_empty():
            pcd = pcd.voxel_down_sample(voxel_size)
            pcds.append(pcd)
        else:
            print(f"Warning: {filename} is empty or could not be loaded.")
    return pcds

def align_point_clouds_sequentially(pcds):
    """ 포인트 클라우드를 순차적으로 정합 (ICP 사용) """
    if len(pcds) < 2:
        raise ValueError("Need at least two point clouds to perform alignment.")
    threshold = 1.0
    trans_init = np.eye(4)
    result = pcds[0]
    for i in range(1, len(pcds)):
        target = pcds[i]
        reg_p2p = o3d.pipelines.registration.registration_icp(
            result, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        result = result.transform(reg_p2p.transformation) + target
    return result

def save_aligned_cloud(aligned_cloud, filename="weSeePJ/saved_pcd/aligned_output.pcd"):
    """ 정합된 포인트 클라우드 저장 """
    if aligned_cloud.is_empty():
        print("Warning: Aligned cloud is empty, not saving.")
        return
    o3d.io.write_point_cloud(filename, aligned_cloud)
    print(f"Aligned point cloud saved as {filename}")
