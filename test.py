from pointcloud.pointcloud import load_point_clouds, align_point_clouds, save_aligned_cloud
from mesh.mesh import create_mesh_from_pcd, save_mesh

def main():
    directory = "weSeePJ/saved_pcd"
    filenames = ["outputCloud_1.pcd", "outputCloud_2.pcd"]

    # PCD 파일 로드
    pcds = load_point_clouds(directory=directory, filenames=filenames)

    # PCD 파일 정합
    aligned_cloud = align_point_clouds(pcds)

    # 결과 저장
    aligned_pcd_filename = f"{directory}/aligned_output.pcd"
    save_aligned_cloud(aligned_cloud, filename=aligned_pcd_filename)

    # 메쉬 생성 및 저장
    mesh = create_mesh_from_pcd(aligned_pcd_filename)
    save_mesh(mesh, filename=f"{directory}/mesh_output.ply")

if __name__ == "__main__":
    main()
