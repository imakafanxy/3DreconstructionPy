import open3d as o3d
import numpy as np

def visualize_point_cloud_with_centered_axis(pcd, window_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    
    # 포인트 클라우드의 중심 계산
    center = pcd.get_center()
    
    # 좌표축 생성 및 중심으로 이동
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=center)
    vis.add_geometry(axis)
    
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

# # 포인트 클라우드 로드
# pcd = o3d.io.read_point_cloud("test_pcd2/front_output.pcd")
# visualize_point_cloud_with_centered_axis(pcd, "Front Point Cloud")

pcd2 = o3d.io.read_point_cloud("C:/Users/iilab/Desktop/성공/front_aligned_output.pcd")
visualize_point_cloud_with_centered_axis(pcd2, "Back Point Cloud")