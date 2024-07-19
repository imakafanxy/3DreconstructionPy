from pointcloud.pointcloud import load_point_clouds, multiway_registration, optimize_pose_graph, apply_pose_graph, save_aligned_cloud, colored_icp_registration
from mesh.mesh import create_mesh_from_pcd, save_mesh
import open3d as o3d

def main():
    directory = "weSeePJ/saved_pcd"
    filenames = [f"outputCloud_{i}.pcd" for i in range(1, 11)]

    # PCD 파일 로드
    pcds = load_point_clouds(directory=directory, filenames=filenames, voxel_size=0.01)

    # PCD 파일 정합
    pose_graph = multiway_registration(pcds, voxel_size=0.01)
    optimized_pose_graph = optimize_pose_graph(pose_graph)
    aligned_cloud = apply_pose_graph(pcds, optimized_pose_graph)

    # ICP를 이용한 세밀 정합
    for i in range(1, len(pcds)):
        icp_result = colored_icp_registration(pcds[i - 1], pcds[i], voxel_size=0.01)
        pcds[i].transform(icp_result.transformation)

    aligned_cloud = o3d.geometry.PointCloud()
    for pcd in pcds:
        aligned_cloud += pcd

    # 결과 저장
    aligned_pcd_filename = f"{directory}/aligned_output.pcd"
    save_aligned_cloud(aligned_cloud, filename=aligned_pcd_filename)

    # 메쉬 생성 및 저장
    mesh = create_mesh_from_pcd(aligned_pcd_filename)
    save_mesh(mesh, filename=f"{directory}/mesh_output.ply")

if __name__ == "__main__":
    main()
