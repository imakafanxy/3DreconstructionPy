import os
import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

class RegistrationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Point Cloud Registration")

        self.source_pcd = None
        self.target_pcd = None
        self.source_points = []
        self.target_points = []

        # UI 요소 설정
        self.load_source_button = tk.Button(master, text="Load Source PCD", command=self.load_source_pcd)
        self.load_source_button.pack()

        self.load_target_button = tk.Button(master, text="Load Target PCD", command=self.load_target_pcd)
        self.load_target_button.pack()

        self.pick_points_button = tk.Button(master, text="Pick Points from Source", command=self.pick_source_points, state=tk.DISABLED)
        self.pick_points_button.pack()

        self.register_button = tk.Button(master, text="Register with Target", command=self.register_pcds, state=tk.DISABLED)
        self.register_button.pack()

        self.save_button = tk.Button(master, text="Save Result", command=self.save_result_pcd, state=tk.DISABLED)
        self.save_button.pack()

    def load_source_pcd(self):
        file_path = filedialog.askopenfilename(filetypes=[("Point Cloud Files", "*.pcd")])
        if file_path:
            self.source_pcd = o3d.io.read_point_cloud(file_path)
            messagebox.showinfo("Info", f"Loaded source PCD: {os.path.basename(file_path)}")
            self.pick_points_button.config(state=tk.NORMAL)

    def load_target_pcd(self):
        file_path = filedialog.askopenfilename(filetypes=[("Point Cloud Files", "*.pcd")])
        if file_path:
            self.target_pcd = o3d.io.read_point_cloud(file_path)
            messagebox.showinfo("Info", f"Loaded target PCD: {os.path.basename(file_path)}")
            if self.source_points:
                self.register_button.config(state=tk.NORMAL)

    def pick_source_points(self):
        if self.source_pcd:
            self.source_points = self.pick_points_from_pcd(self.source_pcd)
            if len(self.source_points) > 0:
                messagebox.showinfo("Info", f"Picked {len(self.source_points)} points from source PCD.")
                if self.target_pcd:
                    self.register_button.config(state=tk.NORMAL)

    def pick_points_from_pcd(self, pcd):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # 사용자 인터페이스 시작
        vis.destroy_window()
        return vis.get_picked_points()

    def register_pcds(self):
        if not self.source_pcd or not self.target_pcd:
            messagebox.showwarning("Warning", "Please load both source and target PCDs.")
            return

        # 타겟 PCD에서 특징점 선택
        self.target_points = self.pick_points_from_pcd(self.target_pcd)
        if len(self.source_points) != len(self.target_points):
            messagebox.showwarning("Warning", "The number of points picked in both PCDs should match.")
            return

        # 선택된 포인트를 기준으로 정합
        source_points_np = np.asarray([self.source_pcd.points[i] for i in self.source_points])
        target_points_np = np.asarray([self.target_pcd.points[i] for i in self.target_points])

        # 포인트 기반 정합 수행
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        transformation = p2p.compute_transformation(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points_np)),
                                                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points_np)))
        
        self.source_pcd.transform(transformation)
        self.result_pcd = self.source_pcd + self.target_pcd

        # 결과 시각화
        o3d.visualization.draw_geometries([self.result_pcd], window_name="Registration Result")

        self.save_button.config(state=tk.NORMAL)

    def save_result_pcd(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".pcd", filetypes=[("Point Cloud Files", "*.pcd")])
        if save_path:
            o3d.io.write_point_cloud(save_path, self.result_pcd)
            messagebox.showinfo("Info", f"Result saved to: {save_path}")

def main():
    root = tk.Tk()
    app = RegistrationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
