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
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    vertex_colors = []
    default_color = [0.5, 0.5, 0.5]  # 기본 색상 (회색)
    for vertex in mesh.vertices:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
        if len(idx) > 0 and idx[0] < len(pcd.colors):
            vertex_colors.append(pcd.colors[idx[0]])
        else:
            vertex_colors.append(default_color)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh

def inpaint_vertex_colors(mesh, neighbors=3):
    print("Inpainting vertex colors...")
    mesh_tree = o3d.geometry.KDTreeFlann(mesh)
    for i, color in enumerate(mesh.vertex_colors):
        if np.array_equal(color, [0.0, 0.0, 0.0]):
            [_, idx, _] = mesh_tree.search_knn_vector_3d(mesh.vertices[i], neighbors)
            neighbor_colors = [mesh.vertex_colors[j] for j in idx if not np.array_equal(mesh.vertex_colors[j], [0.5, 0.5, 0.5])]
            if len(neighbor_colors) > 0:
                mesh.vertex_colors[i] = np.mean(neighbor_colors, axis=0)
    return mesh

def smooth_mesh(mesh, num_iterations=50, lambda_filter=0.5):
    print("Smoothing mesh...")
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=num_iterations, lambda_filter=lambda_filter, filter_scope=o3d.geometry.FilterScope.All)
    mesh.compute_vertex_normals()
    return mesh

def save_mesh(mesh, filename):
    print(f"Saving mesh to {filename}")
    success = o3d.io.write_triangle_mesh(filename, mesh)
    if success:
        print(f"Mesh saved successfully to {filename}")
    else:
        print(f"Failed to save mesh to {filename}")

def visualize_mesh(mesh):
    print("Visualizing mesh...")
    o3d.visualization.draw_geometries([mesh])

def main():
    input_pcd_file = "saved_pcd/성공.pcd"
    output_mesh_file = "saved_pcd/front_aligned_mesh_colored.ply"
    
    pcd = load_point_cloud(input_pcd_file)
    pcd = estimate_normals(pcd)
    
    mesh, densities = poisson_mesh(pcd)
    mesh = remove_low_density_vertices(mesh, densities)
    
    mesh = smooth_mesh(mesh)  # 매쉬 스무딩
    
    mesh = apply_texture_to_mesh(mesh, pcd)
    
    mesh = inpaint_vertex_colors(mesh)  # 색상 보간
    
    save_mesh(mesh, output_mesh_file)
    
    visualize_mesh(mesh)

if __name__ == "__main__":
    main()
