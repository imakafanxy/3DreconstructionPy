import open3d as o3d

def create_mesh_from_pcd(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    if not pcd.has_normals():
        pcd.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    return mesh

def save_mesh(mesh, filename="weSeePJ/saved_pcd/mesh_output.ply"):
    o3d.io.write_triangle_mesh(filename, mesh)
    print(f"Mesh saved as {filename}")