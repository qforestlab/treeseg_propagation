import os
import glob

import open3d as o3d
import numpy as np
import laspy


def read_las_np(pc_path):
    point_cloud = laspy.read(pc_path)
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    return points

def get_plot_mask(plot_file, trees_folder, distance_th=0.01):
    plot_points = read_las_np(plot_file)
    plot_o3d = o3d.geometry.PointCloud()
    plot_o3d.points = o3d.utility.Vector3dVector(plot_points)

    plot_mask = np.ones(plot_points.shape[0], dtype=bool)

    tree_files = glob.glob(os.path.join(trees_folder, "*.las"))

    for i, tree in enumerate(tree_files):
        print(f"Processing {tree} ({i+1}/{len(tree_files)})")

        tree_points = read_las_np(tree)
        tree_o3d = o3d.geometry.PointCloud()
        tree_o3d.points = o3d.utility.Vector3dVector(tree_points)
        
        # get plot points in bbox to limit calculations
        tree_bbox = tree_o3d.get_axis_aligned_bounding_box()
        # add small buffer around tree
        tree_bbox.max_bound = tree_bbox.max_bound + np.array([distance_th+0.01, distance_th+0.01, distance_th+0.01])
        tree_bbox.min_bound = tree_bbox.min_bound - np.array([distance_th+0.01, distance_th+0.01, distance_th+0.01])

        inliers_indices = tree_bbox.get_point_indices_within_bounding_box(plot_o3d.points)
        inliers_pc = plot_o3d.select_by_index(inliers_indices, invert=False)

        # for points in bbox, calculate distance to tree and get their indices
        distances = inliers_pc.compute_point_cloud_distance(tree_o3d)
        distances = np.asarray(distances)
        tree_ind = np.where(distances < distance_th)[0]

        # label original instance array
        inliers_indices = np.asarray(inliers_indices)
        plot_indices_tree = inliers_indices[tree_ind] # map mask on inlier indices to mask on original indices

        plot_mask[plot_indices_tree] = 0

    return plot_mask


def segment_plot(plot_file, mask, opath="plot_propagated.las"):
    point_cloud = laspy.read(plot_file)
    mask = np.array(mask, dtype=bool)

    plot_points_left = point_cloud.points[mask].copy()

    output_file = laspy.LasData(point_cloud.header)
    output_file.points = plot_points_left
    output_file.write(opath)
    return


def main():
    # change paths:
    # file with original plot point cloud
    plot_file = "/Stor1/wout/data/Gontrode/2023-03-09_ForSe_Gontrode_all.las"
    # folder containing .las files of individual trees
    trees_folder = "/Stor1/wout/data/Gontrode/trees"
    # output path for propagated pointcloud
    opath = "/Stor1/wout/data/Gontrode/plot_propagated.las"

    # change this for more rough/fine segmentation
    distance_th = 0.001

    plot_mask = get_plot_mask(plot_file, trees_folder, distance_th)

    # cache for if there are issues with step 2
    plot_mask = np.save("mask_plot.npy")
    # uncomment if you want to use saved mask
    # plot_mask = np.load("mask_plot.npy")

    # seperated mask propagation and actual application so hopefully memory can be cleared up
    segment_plot(plot_file, plot_mask, opath)

if __name__ == "__main__":
    main()