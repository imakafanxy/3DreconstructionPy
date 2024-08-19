import open3d as o3d
import numpy as np

def load_point_cloud(filename):
    print(f"Loading point cloud from {filename}")
    pcd = o3d.io.read_point_cloud(filename)
    if pcd.is_empty():
        raise ValueError("Point cloud is empty.")
    print(f"Point cloud loaded with {len(pcd.points)} points.")
    return pcd

def estimate_normals(pcd, search_radius=0.1):
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30))
    return pcd

def poisson_mesh(pcd, depth=9):
    print("Creating Poisson mesh...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    return mesh, densities

def remove_low_density_vertices(mesh, densities, threshold=0.01):
    print("Removing low density vertices...")
    vertices_to_remove = densities < np.quantile(densities, threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh

def apply_texture_to_mesh(mesh, pcd):
    print("Applying texture to mesh...")
    mesh.vertex_colors = pcd.colors
    return mesh

def save_mesh(mesh, filename):
    print(f"Saving mesh to {filename}")
    success = o3d.io.write_triangle_mesh(filename, mesh)
    if success:
        print(f"Mesh saved successfully to {filename}")
    else:
        print(f"Failed to save mesh to {filename}")

def main():
    input_pcd_file = "saved_pcd/front_aligned_output.pcd"
    output_mesh_file = "saved_pcd/front_aligned_mesh.ply"
    
    pcd = load_point_cloud(input_pcd_file)
    pcd = estimate_normals(pcd)
    
    mesh, densities = poisson_mesh(pcd)
    mesh = remove_low_density_vertices(mesh, densities)
    
    mesh = apply_texture_to_mesh(mesh, pcd)
    
    save_mesh(mesh, output_mesh_file)

if __name__ == "__main__":
    main()
