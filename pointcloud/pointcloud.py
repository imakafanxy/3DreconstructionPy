# import open3d as o3d
# import numpy as np

# def load_point_clouds(directory="saved_pcd", num_files=8):
#     pcds = []
#     for i in range(num_files):
#         pcd = o3d.io.read_point_cloud(f"{directory}/outputCloud_{i}.pcd")
#         pcds.append(pcd)
#     return pcds

# def align_point_clouds(pcds):
#     threshold = 1.0
#     trans_init = np.eye(4)
#     icp_results = []
#     source = pcds[0]
#     for i in range(1, len(pcds)):
#         target = pcds[i]
#         reg_p2p = o3d.pipelines.registration.registration_icp(
#             source, target, threshold, trans_init,
#             o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#             o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
#         )
#         source.transform(reg_p2p.transformation)
#         icp_results.append(source)
#     return icp_results

# def save_aligned_cloud(aligned_cloud, filename="aligned_output.pcd"):
#     o3d.io.write_point_cloud(filename, aligned_cloud)
#     print(f"Aligned point cloud saved as {filename}")

# # if __name__ == "__main__":
# #     pcds = load_point_clouds()
# #     aligned_cloud = align_point_clouds(pcds)
# #     save_aligned_cloud(aligned_cloud)

import open3d as o3d
import numpy as np

def load_point_clouds(directory="weSeePJ/saved_pcd", filenames=None, voxel_size=0.02):
    if filenames is None:
        filenames = ["outputCloud_1.pcd", "outputCloud_2.pcd"]
    pcds = []
    for filename in filenames:
        pcd = o3d.io.read_point_cloud(f"{directory}/{filename}")
        if not pcd.is_empty():
            pcd = pcd.voxel_down_sample(voxel_size)
            pcds.append(pcd)
        else:
            print(f"Warning: {filename} is empty or could not be loaded.")
    return pcds

def align_point_clouds(pcds):
    if len(pcds) < 2:
        raise ValueError("Need at least two point clouds to perform alignment.")
    threshold = 1.0
    trans_init = np.eye(4)
    source = pcds[0]
    for i in range(1, len(pcds)):
        target = pcds[i]
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        source.transform(reg_p2p.transformation)
    return source

def save_aligned_cloud(aligned_cloud, filename="weSeePJ/aligned_output.pcd"):
    if aligned_cloud.is_empty():
        print("Warning: Aligned cloud is empty, not saving.")
        return
    o3d.io.write_point_cloud(filename, aligned_cloud)
    print(f"Aligned point cloud saved as {filename}")
