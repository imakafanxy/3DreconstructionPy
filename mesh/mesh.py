import open3d as o3d
import numpy as np

def create_mesh_from_pcd(pcd_filename):
    try:
        print(f"Loading point cloud from {pcd_filename} for mesh creation.")
        pcd = o3d.io.read_point_cloud(pcd_filename)
        print("Point cloud loaded successfully.")
        
        # Check if point cloud is empty
        if pcd.is_empty():
            print("Error: Point cloud is empty.")
            return None
        
        print("Estimating normals for the point cloud.")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        
        print("Performing Poisson surface reconstruction.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        print("Poisson surface reconstruction completed.")
        
        print("Filtering low density vertices.")
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print("Mesh creation successful.")
        return mesh
    except Exception as e:
        print(f"Exception during mesh creation: {e}")
        return None

def save_mesh(mesh, filename="weSeePJ/saved_pcd/mesh_output.ply"):
    """ 생성된 메쉬 저장 """
    o3d.io.write_triangle_mesh(filename, mesh)
    print(f"Mesh saved as {filename}")
