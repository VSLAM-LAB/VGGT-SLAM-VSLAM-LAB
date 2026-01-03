import os
import glob
import argparse

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import yaml
import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver

from vggt.models.vggt import VGGT


parser = argparse.ArgumentParser(description="VGGT-SLAM demo")

parser.add_argument("--sequence_path", type=Path, required=True)
parser.add_argument("--calibration_yaml", type=Path, required=True)
parser.add_argument("--rgb_csv", type=Path, required=True)
parser.add_argument("--exp_folder", type=Path, required=True)
parser.add_argument("--exp_it", type=str, default="0")
parser.add_argument("--settings_yaml", type=Path, default=None)
parser.add_argument("--verbose", type=str, help="verbose")


parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--use_sim3", action="store_true", help="Use Sim3 instead of SL(4)")
parser.add_argument("--plot_focal_lengths", action="store_true", help="Plot focal lengths for the submaps")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--downsample_factor", type=int, default=1, help="Factor to reduce image size by 1/N")
parser.add_argument("--max_loops", type=int, default=1, help="Maximum number of loop closures per submap")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--vis_stride", type=int, default=1, help="Stride interval in the 3D point cloud image for visualization. Try increasing (such as 4) to reduce lag in visualizing large maps.")
parser.add_argument("--vis_point_size", type=float, default=0.003, help="Visualization point size")

def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args, _ = parser.parse_known_args()
    args.vis_map = bool(int(args.verbose))

    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        use_sim3=args.use_sim3,
        gradio_mode=False,
        vis_stride = args.vis_stride,
        vis_point_size = args.vis_point_size,
    )

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(device)

    # Use the provided image folder path
    df = pd.read_csv(args.rgb_csv)       
    with open(args.settings_yaml, 'r') as f:
        settings = yaml.safe_load(f)
    cam_name = settings.get('cam_mono', "rgb_0")

    image_list = df[f'path_{cam_name}'].to_list()
    ts_ns = df[f'ts_{cam_name} (ns)'].to_list()
    image_names = []
    for i, imrel in enumerate(image_list):
        image_names.append(args.sequence_path / imrel)

    print(f"Loading images from {args.sequence_path}...")
    image_names = utils.sort_images_by_number(image_names)
    image_names = utils.downsample_images(image_names, args.downsample_factor)
    ts_ns = utils.downsample_images(ts_ns, args.downsample_factor)
    print(f"Found {len(image_names)} images")
    print(f"Found {len(ts_ns)} timestamps")
    
    image_names_subset = []
    timestamps_subset = []
    data = []
    for i, image_name in enumerate(tqdm(image_names)):
        if use_optical_flow_downsample:
            img = cv2.imread(image_name)
            enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
            if enough_disparity:
                image_names_subset.append(image_name)
                timestamps_subset.append(ts_ns[i])
        else:
            image_names_subset.append(image_name)
            timestamps_subset.append(ts_ns[i])

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            print(image_names_subset)
            print(len(image_names_subset),len(timestamps_subset))
            
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops, timestamps = timestamps_subset)

            data.append(predictions["intrinsic"][:,0,0])

            solver.add_points(predictions)

            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
            
            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]
            timestamps_subset = timestamps_subset[-args.overlapping_window_size:]

    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    keyframe_csv = args.exp_folder / f"{args.exp_it.zfill(5)}_KeyFrameTrajectory.csv"
    solver.map.write_poses_to_file_vslamlab(keyframe_csv)

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.plot_focal_lengths:
        # Define a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        for i, values in enumerate(data):
            y = values  # Y-values from the list
            x = [i] * len(values)  # X-values (same for all points in the list)
            plt.scatter(x, y, color=colors[i], label=f'List {i+1}')

        plt.xlabel("poses")
        plt.ylabel("Focal lengths")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()
