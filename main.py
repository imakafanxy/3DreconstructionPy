# from camera import start_camera, stop_camera, capture_images
# from pointcloud import load_point_clouds, align_point_clouds, save_aligned_cloud
# from mesh import create_mesh_from_pcd, save_mesh

# def main():
#     pipeline, profile = start_camera()
#     try:
#         capture_images(pipeline)
#     finally:
#         stop_camera(pipeline)
    
#     pcds = load_point_clouds()
#     aligned_cloud = align_point_clouds(pcds)
#     save_aligned_cloud(aligned_cloud)
    
#     mesh = create_mesh_from_pcd()
#     save_mesh(mesh)

# if __name__ == "__main__":
#     main()
