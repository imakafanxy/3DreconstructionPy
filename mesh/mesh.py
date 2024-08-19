import open3d as o3d
import numpy as np

def create_mesh_from_pcd(pcd_filename):
    try:
        print(f"Loading point cloud from {pcd_filename} for mesh creation.")
        pcd = o3d.io.read_point_cloud(pcd_filename)
        print("Point cloud loaded successfully.")
        
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
    o3d.io.write_triangle_mesh(filename, mesh)
    print(f"Mesh saved as {filename}")

def create_smooth_mesh_with_texture(pcd):
    print("Starting surface reconstruction...")
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]
    print("Completed Poisson reconstruction.")
    
    print("Simplifying mesh...")
    poisson_mesh = poisson_mesh.simplify_quadric_decimation(100000)
    print("Completed mesh simplification.")
    
    print("Applying mesh filter...")
    poisson_mesh = poisson_mesh.filter_smooth_simple(number_of_iterations=5)
    print("Completed mesh filtering.")
    
    print("Texture mapping...")
    poisson_mesh.compute_vertex_normals()
    poisson_mesh = poisson_mesh.paint_uniform_color([1, 0.706, 0])
    
    return poisson_mesh
