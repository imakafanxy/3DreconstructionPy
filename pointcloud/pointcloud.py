import open3d as o3d
import numpy as np
import os

def load_and_preprocess_point_clouds(directory, filenames=None, voxel_size=0.05, 
                                     radius=0.1, max_nn=50, nb_neighbors=20, std_ratio=2.0, 
                                     distance_threshold=2.0, eps=0.05, min_samples=10):
    """ 포인트 클라우드를 불러오고 전처리를 수행하는 함수 """
    pcds = []
    
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        print(f"Loading file: {filepath}")
        pcd = o3d.io.read_point_cloud(filepath)
        
        if not pcd.is_empty():
            print(f"Point cloud loaded with {len(pcd.points)} points.")
            
            # 1. Voxel downsampling (해상도 조정)
            pcd = pcd.voxel_down_sample(voxel_size)
            print(f"Point cloud downsampled with voxel size {voxel_size}.")
            
            # 2. 노이즈 제거 (Statistical Outlier Removal)
            pcd = remove_outliers(pcd, nb_neighbors, std_ratio)
            print(f"Outliers removed with nb_neighbors={nb_neighbors} and std_ratio={std_ratio}.")
            
            # 3. 노말 추정 (Normal Estimation)
            pcd = estimate_normals(pcd, radius, max_nn)
            print(f"Normals estimated with radius={radius} and max_nn={max_nn}.")
            
            # 4. 배경 제거 (Background Removal)
            pcd = remove_background(pcd, distance_threshold)
            print(f"Background removed with distance threshold={distance_threshold}.")
            
            # 5. 작은 클러스터 제거 (Remove Small Clusters)
            pcd = remove_small_clusters(pcd, eps, min_samples)
            print(f"Small clusters removed with eps={eps} and min_samples={min_samples}.")
            
            # 최종 포인트 클라우드를 리스트에 추가
            pcds.append(pcd)
        else:
            print(f"Warning: {filename} is empty or could not be loaded.")
    
    return pcds


def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """ 아웃라이어를 제거하는 함수 """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = pcd.select_by_index(ind)
    return pcd

def estimate_normals(pcd, radius=0.1, max_nn=50):
    """ 포인트 클라우드의 노말을 추정하는 함수 """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return pcd

def remove_background(pcd, distance_threshold=2.0):
    """ 포인트 클라우드에서 배경을 제거하는 함수 """
    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points, axis=1)
    mask = distances < distance_threshold
    pcd = pcd.select_by_index(np.where(mask)[0])
    return pcd

def remove_small_clusters(pcd, eps=0.05, min_samples=10):
    """ 작은 클러스터를 제거하는 함수 """
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples))
    major_cluster = labels == np.argmax(np.bincount(labels[labels >= 0]))
    return pcd.select_by_index(np.where(major_cluster)[0])

def global_registration(source, target, voxel_size):
    """ 전역 정합 (RANSAC)을 통해 대략적인 정렬을 수행하는 함수 """
    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, o3d.pipelines.registration.FPFHFeature(),
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result_ransac

def icp_refinement(source, target, initial_transformation, max_correspondence_distance_fine):
    """ ICP 기반으로 정밀 정합을 수행하는 함수 """
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return icp_result

def full_registration(pcds, voxel_size, max_correspondence_distance_fine):
    """ 모든 포인트 클라우드를 정합하고 최종 정밀 정합을 수행하는 함수 """
    total_pcd = pcds[0]  # 첫 번째 포인트 클라우드를 기준으로 합침
    transformations = [np.identity(4)]  # 첫 번째 포인트 클라우드의 변환은 기본적으로 단위 행렬

    for i in range(1, len(pcds)):
        print(f"Global registration between point cloud {i} and {i+1}")
        
        # 전역 정합 수행
        result_ransac = global_registration(pcds[i-1], pcds[i], voxel_size)
        print(f"Global registration result between cloud {i} and {i+1}: {result_ransac.transformation}")
        
        # 정밀 정합 수행 (ICP)
        print(f"Refining with ICP between point cloud {i} and {i+1}")
        result_icp = icp_refinement(pcds[i-1], pcds[i], result_ransac.transformation, max_correspondence_distance_fine)
        
        # 누적 변환
        transformations.append(result_icp.transformation)
        
        # 변환 후 정합
        total_pcd = total_pcd.transform(result_icp.transformation)
        total_pcd += pcds[i].transform(result_icp.transformation)
    
    return total_pcd, transformations
