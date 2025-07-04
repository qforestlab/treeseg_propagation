### Tree Instance Propagation to Original Plot Point Cloud 

This repo contains two scripts aimed at removing points present in tree instances from original plot point clouds using distance-based thresholding.  
Given a set of segmented tree point clouds, and the original, unedited plot point cloud, the segmented tree points will be propagated to the plot point cloud and removed. 

---

#### 📄 `treeseg_propagation.py`

This script **creates a propagated point cloud** (i.e., the original plot point cloud without the segmented trees).

**How to run:**
Replace the paths in the `main` function at the bottom of the script and execute the Python file.

**Dependencies:**  
- `open3d`  
- `laspy`  
(Tested using Ubuntu + Python 3.10)

**Notes:**
- The trees must be in the **original coordinate system**.
- With minor adaptations, the same code can be used to **label** (instead of remove) points — e.g., for instance segmentation training/testing data.
- The script reads **`.las` files**. To use `.ply` or `.txt`, write a custom reader function with the same output format as `read_las_np`, and adjust other functions accordingly.
- The algorithm is split into two steps due to potential **memory issues**, especially with undownsampled data.
    - If the function fails in the second step, a **plot mask** is saved as an `.npy` file.
    - You can **load the saved mask** in the `main` function to avoid recomputing.

**Contact:**  
📧 wout.cherlet@ugent.be

---

#### 📄 `tree_extraction.py`

This script creates both:
1. A **propagated point cloud**
2. **Individual tree point clouds** using points from the original plot point cloud.

**How to run:**  
Execute from the command line:

```bash
python tree_extraction_v2.py <input_file.las> <input_directory> <output_directory> <propagated_file.las> --distance_th <value>
```

**Example:**
```bash
python tree_extraction_v2.py /path/to/input.las /path/to/input_dir/ /path/to/output_dir/ /path/to/propagated.las --distance_th 0.015
```

**Dependencies:**  
- `open3d`  
- `laspy`
- laz backends
  - `lazrs` or,
  - `laszip`
- `argparse`

**Notes:**
- The trees must be in the **original coordinate system**.
- The script reads **`.las` and `.laz` files** automatically. It recognizes if you have a laz backend installed.
- As with the other script, the algorithm is split into two steps due to **memory constraints**.
    - If the second step fails, the **plot mask** is saved as an `.npy` file and can be **reloaded later**.

**Contact:**  
📧 geike.desloover@ugent.be  
📧 louise.terryn@ugent.be
