import os
import torch
import numpy as np
import open3d as o3d
from natsort import natsorted
import click
import tqdm
import MinkowskiEngine as ME

def parse_calibration(filename):
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib

def load_poses(calib_fname, poses_fname, kitty360, start_i, end_i):
    if os.path.exists('./Datasets/KITTI-360/calibration/calib_cam_to_velo.txt'):
        calibration = parse_calibration('./Datasets/KITTI-360/calibration/calib_cam_to_velo.txt')
        cam_to_velo = calibration["Tr"]
        velo_to_cam = np.linalg.inv(cam_to_velo)

    if os.path.exists('./Datasets/KITTI-360/calibration/calib_cam_to_pose.txt'):
        calibration = parse_calibration('./Datasets/KITTI-360/calibration/calib_cam_to_pose.txt')
        cam_to_pose = calibration["image_00"]

    poses_file = open(poses_fname)
    poses = []
    avail_index = []

    for line in poses_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        index = int(values[0])
        if index <= start_i or index >= end_i:
            continue
        avail_index.append(index)
        pose[0, 0:4] = values[1:5]
        pose[1, 0:4] = values[5:9]
        pose[2, 0:4] = values[9:13]
        pose[3, 3] = 1.0

        poses.append(pose @ velo_to_cam)

    return poses, avail_index

@click.command()
@click.option('--path', '-p', type=str, help='path to the scan sequence')
@click.option('--voxel_size', '-v', type=float, default=0.1, help='voxel size')
@click.option('--cpu', '-c', is_flag=True, help='Use CPU')
def main(path, voxel_size, cpu):
    device_label = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_label = 'cpu' if cpu else device_label
    device = torch.device(device_label)
    # for seq in ['00','01','02','03','04','05','06','07','08','09','10']:
    map_points = torch.empty((0,3)).to(device)

    poses, avail_seq = load_poses('', './Datasets/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/cam0_to_world.txt', True, 3000, 7000)
    for pose, seq in tqdm.tqdm(list(zip(poses, avail_seq))):
        pose = torch.from_numpy(pose).float().to(device)
        pcd_file = f'./Datasets/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/{str(seq).zfill(10)}.bin'
        points = torch.from_numpy(np.fromfile(pcd_file, dtype=np.float32)).to(device)
        points = points.reshape(-1,4)

        # label_file = pcd_file.replace('velodyne', 'labels').replace('.bin', '.label')
        # l_set = np.fromfile(label_file, dtype=np.uint32)
        # l_set = l_set.reshape((-1))
        # l_set = l_set & 0xFFFF

        # remove moving points
        # static_idx = (l_set < 252) & (l_set > 1)
        # points = points[static_idx]

        # remove flying artifacts
        dist = torch.pow(points, 2)
        dist = torch.sqrt(dist.sum(-1))
        points = points[dist > 3.5]

        points[:,-1] = 1.
        points = points @ pose.T

        map_points = torch.cat((map_points, points[:,:3]), axis=0)
        _, mapping = ME.utils.sparse_quantize(coordinates=map_points / voxel_size, return_index=True, device=device_label)
        map_points = map_points[mapping]


    print(f'saving map')
    np.save('./Datasets/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/map_clean.npy', map_points.cpu().numpy())


if __name__ == '__main__':
    main()
