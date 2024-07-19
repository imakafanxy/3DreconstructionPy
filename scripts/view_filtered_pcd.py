import open3d as o3d
import numpy as np

def filter_pcd_by_distance(pcd, max_distance=1.0):
    """포인트 클라우드를 주어진 거리 내로 필터링"""
    distances = np.asarray(pcd.points)[:, 2]  # Z축 거리
    mask = distances < max_distance
    pcd_filtered = pcd.select_by_index(np.where(mask)[0])
    return pcd_filtered

# Example usage with the aligned output PCD
if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("weSeePJ/saved_pcd/aligned_output.pcd")
    pcd_filtered = filter_pcd_by_distance(pcd)
    o3d.visualization.draw_geometries([pcd_filtered])
