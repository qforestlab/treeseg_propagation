### Tree instance propagation to original plot point cloud

This repo contains a single script, aimed at removing points present in tree instances from the original plot point clouds using a distance-based thresholding.
To run, simple replace the paths in the main function at the bottom of the scripts and execute the python file.

Dependencies: open3d, laspy (tested using ubuntu+python 3.10)

Some notes:
- The trees must be in the original coordinate system
- With small adaptations, the same code could be used to label instead of remove the points from the plot, for e.g. instance segmentation training/testing data
- The scripts reads las files, to use ply or txt just write a function with the same output types as the read_las_np and adapt the other functions a bit
- The algorithm is divided into two functions, as memory issues might come up, especially with undownsampled data. If the function fails in the second step, the plot mask is saved as an .npy file, and you can load it back in (in the main function) without rerunning all the calculations

Questions: wout.cherlet@ugent.be
