import os
import glob
import open3d as o3d
import numpy as np
import laspy
import argparse

def get_file_extension(file_path):
    """
    Returns the file extension for the given file path in lowercase.
    If the file has no extension, returns an empty string.
    """
    _, extension = os.path.splitext(file_path)
    if len(extension) == 0:
        extension = os.listdir(file_path)[0][-4:]
    return extension.lower()

def get_laz_backend():
    # Try using the Lazrs backend first
    backend = laspy.LazBackend.Lazrs
    if backend.is_available():
        print("Using lazrs backend for LAZ files.")
        return backend
    # Fallback: try the Laszip backend
    backend = laspy.LazBackend.Laszip
    if backend.is_available():
        print("Using laszip backend for LAZ files.")
        return backend
    raise Exception("No LAZ backend available. Please install lazrs or laszip.")

LAZ_BACKEND = get_laz_backend()

def read_las_np(pc_path, laz_backend=LAZ_BACKEND):
    ext = get_file_extension(pc_path)
    if ext == ".las":
        point_cloud = laspy.read(pc_path)
    elif ext == ".laz":
        point_cloud = laspy.read(pc_path, laz_backend=laz_backend)
    else:
        raise ValueError(f"Unsupported file extension: {ext} for file {pc_path}")
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    return points, point_cloud

def get_plot_mask(plot_file, trees_folder, opath_folder, distance_th=0.01):
    ext = get_file_extension(plot_file)
    plot_points, plot_cloud = read_las_np(plot_file)
    plot_o3d = o3d.geometry.PointCloud()
    plot_o3d.points = o3d.utility.Vector3dVector(plot_points)

    plot_mask = np.ones(plot_points.shape[0], dtype=bool)
    ext = get_file_extension(trees_folder)
    tree_files = glob.glob(os.path.join(trees_folder, "*" + ext))
    
    for i, tree in enumerate(tree_files):
        print(f"Processing {tree} ({i+1}/{len(tree_files)})")
        tree_mask = np.zeros(plot_points.shape[0], dtype=bool)
        ext = get_file_extension(tree)
        tree_points, tree_cloud = read_las_np(tree)       
        tree_o3d = o3d.geometry.PointCloud()
        tree_o3d.points = o3d.utility.Vector3dVector(tree_points)
        
        # Get plot points in the tree's bounding box (with a small buffer)
        tree_bbox = tree_o3d.get_axis_aligned_bounding_box()
        buffer = np.array([distance_th + 0.01, distance_th + 0.01, distance_th + 0.01])
        tree_bbox.max_bound = tree_bbox.max_bound + buffer
        tree_bbox.min_bound = tree_bbox.min_bound - buffer

        inliers_indices = tree_bbox.get_point_indices_within_bounding_box(plot_o3d.points)
        inliers_pc = plot_o3d.select_by_index(inliers_indices, invert=False)
        
        if len(inliers_pc.points) != 0:
            # For points in the bounding box, compute distances and update the mask
            distances = inliers_pc.compute_point_cloud_distance(tree_o3d)
            distances = np.asarray(distances)
            tree_ind = np.where(distances < distance_th)[0]
            # Tree indices in the plot cloud
            inliers_indices = np.asarray(inliers_indices)
            plot_indices_tree = inliers_indices[tree_ind]
            # Write tree to seperate folder
            tree_mask[plot_indices_tree] = True
            tree_mask_noduplicates = tree_mask & plot_mask
            tree_points = plot_cloud.points[tree_mask_noduplicates].copy()
            tree_output_file = laspy.LasData(plot_cloud.header)
            tree_output_file.points = tree_points
            tree_output_filename = os.path.basename(tree)[:-4] + "_propagated.las"
            tree_output_path = os.path.join(opath_folder, tree_output_filename)
            tree_output_file.write(tree_output_path)
            # Update plot mask
            plot_mask[plot_indices_tree] = False


    return plot_mask

def segment_plot(plot_file, mask, opath):
    ext = get_file_extension(plot_file)
    plot_points, plot_cloud = read_las_np(plot_file)
    mask = np.array(mask, dtype=bool)
    plot_points_left = plot_cloud.points[mask].copy()
    output_file = laspy.LasData(plot_cloud.header)
    output_file.points = plot_points_left
    output_file.write(opath)
    return

def main():

    parser = argparse.ArgumentParser(
        description="Process file paths for plot and trees."
    )
    # Define expected command-line arguments
    parser.add_argument(
        "plot_file",
        type=str,
        help="Path to the plot file"
    )
    parser.add_argument(
        "trees_folder",
        type=str,
        help="Path to the folder containing tree files"
    )
    parser.add_argument(
        "opath_folder",
        type=str,
        help="Path to the output folder"
    )
    parser.add_argument(
        "opath_plot",
        type=str,
        help="Path to the output plot file"
    )
    parser.add_argument(
        "--distance_th",
        type=float,
        default=0.01,
        help="Distance threshold for tree points to be considered part of the plot"
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Assign variables from parsed arguments
    plot_file = args.plot_file
    trees_folder = args.trees_folder
    opath_folder = args.opath_folder
    opath_plot = args.opath_plot

    os.makedirs(opath_folder, exist_ok=True)

    distance_th = args.distance_th
    plot_mask = get_plot_mask(plot_file, trees_folder, opath_folder, distance_th)
    
    # Cache the mask if needed
    np.save("mask_plot.npy", plot_mask)
    # Uncomment to load a saved mask:
    plot_mask = np.load("mask_plot.npy")
    
    segment_plot(plot_file, plot_mask, opath_plot)

if __name__ == "__main__":
    main()

