import open3d as o3d
import numpy as np
import click
        
                                                                                                                                     
@click.command()
@click.option('--path', '-p', type=str, help='path to pcd')
def main(path):
    pcd = o3d.io.read_point_cloud(path)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=5.0)
    pcd = pcd.select_by_index(ind)

    pcd.estimate_normals()
    
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.4,
                                      front=[-1, 1, 1],
                                      lookat=[0, 0, 0],
                                      up=[0, 0, 1])
                                                                                                                                     
if __name__ == '__main__':
    main()
