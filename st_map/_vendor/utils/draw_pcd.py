# @brief:  通过保存出来的点云，对场景进行点云重建，然后将其中识别出来的物体通过不同颜色的方式标出来。
import os, sys
import open3d as o3d

import os, sys
import colorsys
import numpy as np
import pickle as pkl
from st_map._vendor.slam.utils import extractVoxelCenterfromVoxelmap,selectPointsFromVoxel,selectPcdFromVoxel
from detectron2.data import MetadataCatalog
import time
import matplotlib.colors as mcolors
from st_map._vendor.utils.vis import LineMesh
import argparse
# metadata = MetadataCatalog.get('coco_2017_train_panoptic')
css4_colors = mcolors.CSS4_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]
def create_ball_mesh(center, radius, color=(0, 1, 0)):
    """
    Create a colored mesh sphere.
    
    Args:
    - center (tuple): (x, y, z) coordinates for the center of the sphere.
    - radius (float): Radius of the sphere.
    - color (tuple): RGB values in the range [0, 1] for the color of the sphere.
    
    Returns:
    - o3d.geometry.TriangleMesh: Colored mesh sphere.
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere
def read_pkl(path_file):
    output = open(path_file, 'rb')
    res = pkl.load(output)

    for obj in res['objects']:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(obj['pcd']['points'])
        point_cloud.colors = o3d.utility.Vector3dVector(obj['pcd']['colors'])
        obj['pcd'] = point_cloud

        if 'min_bound' in obj['bbox']:
            obj['bbox'] = o3d.geometry.AxisAlignedBoundingBox(
                min_bound = obj['bbox']['min_bound'],
                max_bound = obj['bbox']['max_bound']
            )
        elif 'center' in obj['bbox']:
            obj['bbox'] = o3d.geometry.OrientedBoundingBox(
                center = obj['bbox']['center'],
                R = obj['bbox']['R'],
                extent = obj['bbox']['extent']
            )

    for obj in res['bg_objects']:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(res['bg_objects'][obj]['pcd']['points'])
        point_cloud.colors = o3d.utility.Vector3dVector(res['bg_objects'][obj]['pcd']['colors'])
        res['bg_objects'][obj]['pcd'] = point_cloud

        if 'min_bound' in res['bg_objects'][obj]['bbox']:
            res['bg_objects'][obj]['bbox'] = o3d.geometry.AxisAlignedBoundingBox(
                min_bound = res['bg_objects'][obj]['bbox']['min_bound'],
                max_bound = res['bg_objects'][obj]['bbox']['max_bound']
            )
        elif 'center' in res['bg_objects'][obj]['bbox']:
            res['bg_objects'][obj]['bbox'] = o3d.geometry.OrientedBoundingBox(
                center = res['bg_objects'][obj]['bbox']['center'],
                R = res['bg_objects'][obj]['bbox']['R'],
                extent = res['bg_objects'][obj]['bbox']['extent']
            )

    return res

def draw_voxel_objects(objects,shape,min_bound,idx=None,pcd_path=None):
    pcd = o3d.geometry.PointCloud()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="test_poses_with_pcd")
    points=np.empty([0,3])
    normals=np.empty([0,3])
    colors=np.empty([0,3])
    for i, obj in enumerate(objects):
        coordinates = extractVoxelCenterfromVoxelmap(obj['voxel_index'],shape,min_bound,0.25)
        points = np.concatenate((points,coordinates),axis=0)
        normals = np.concatenate((normals,np.ones_like(coordinates)),axis=0)
        colors = np.concatenate((colors,np.ones_like(coordinates)*color_proposals[i%len(color_proposals)]),axis=0)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(pcd_path+"sem_"+str(idx)+".ply", pcd)
def drawScenegraph(objects,bg_objects,edges,shape,min_bound,name=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="scene",width=1280,height=960,left=100)

    # Load edge files and create meshes for the scene graph
    
    colors = [color_proposals[c] for c in range(len(objects))]
    obj_centers = []
    scene_graph_geometries = []
    for i in range(len(objects)):
        pcd = selectPcdFromVoxel(objects[i],0.25,shape,min_bound)
        points = np.asarray(pcd.points)
        center = np.mean(points, axis=0)
        # radius = extent ** 0.5 / 25
        radius = 0.1
        obj_centers.append(center)
        # remove the nodes on the ceiling, for better visualization
        ball = create_ball_mesh(center, radius, colors[i])

        pcd.normals=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points)))
        pcd.colors=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points))*colors[i])
        # pcd.colors=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points))*colors[i])
        # o3d.visualization.draw_geometries([pcd])
        vis.add_geometry(pcd)
        scene_graph_geometries.append(ball)
    for key,value in bg_objects.items():
        if value is None or key == "ceiling" :
            continue
        pcd = selectPcdFromVoxel(value,0.25,shape,min_bound)
        points = np.asarray(pcd.points)
        center = np.mean(points, axis=0)
        # radius = extent ** 0.5 / 25
        # radius = 8
        # obj_centers.append(center)
        # # remove the nodes on the ceiling, for better visualization
        # ball = create_ball_mesh(center, radius, colors[i])

        pcd.normals=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points)))
        pcd.colors=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points))*[211/255,211/255,211/255])
        # pcd.colors=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points))*colors[i])
        # o3d.visualization.draw_geometries([pcd])
        vis.add_geometry(pcd)
        # scene_graph_geometries.append(ball)
    # vis.capture_screen_image("connection.jpg")
    for id1,value in edges.items():
        for id2 in value.keys():
            if id2 in edges.keys():
                line_mesh = LineMesh(
                            points = np.array([obj_centers[int(id1)], obj_centers[int(id2)]]),
                            lines = np.array([[0, 1]]),
                            colors = [250/255,109/255,29/255],
                            radius=0.01
                        )
               
                scene_graph_geometries.extend(line_mesh.cylinder_segments)
            
        for geometry in scene_graph_geometries:
            vis.add_geometry(geometry, reset_bounding_box=False)
    # vis.capture_screen_image("connection.jpg")
    vis.run()
    # vis.poll_events()
    # vis.update_renderer()
    # time.sleep(2)
    # if name is not None :
    #     # vis.update_renderer()
    #     vis.capture_screen_image(name)
    vis.destroy_window()
def save_image(objects, bg_objects, edges, save_path, camera_pose, base_name,shape,min_bound):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="scene", width=1280, height=960)
    # remove_list = [47, 51, 63, 59, 31, 26, 68, 65, 60, 61]
    remove_list = []

    # Load edge files and create meshes for the scene graph

    colors = [color_proposals[c] for c in range(len(objects))]
    objj_centers = []
    scene_graph_geometries = []
    combined_pcd = o3d.geometry.PointCloud()  # 空的点云
    combined_mesh = o3d.geometry.TriangleMesh()  # 空的三角网格
    obj_pcd = o3d.geometry.PointCloud()
    bg_pcd = o3d.geometry.PointCloud()

    for i in range(len(objects)):
        # print(colors[i], ",")
            pcd = selectPcdFromVoxel(objects[i],0.25,shape,min_bound)
            # pcd = obj_centers[i]['pcd']
            # cl, ind = pcd.remove_radius_outlier(nb_points=50, radius=2)
            # inlier_cloud = pcd.select_by_index(ind)
            points = np.asarray(pcd.points)
            index = np.where((points[:, 2] < 2.5) & (points[:, 2] > 0.08))[0]
            points = points[index]
            center = np.mean(points, axis=0)
            # radius = extent ** 0.5 / 25
            radius = 0.1
            objj_centers.append(center)
            # remove the nodes on the ceiling, for better visualization
            if i not in remove_list and len(points) > 20 and objects[i]['exists_in_SG']:
                # ball = create_ball_mesh(center, radius, colors[i])
                # scene_graph_geometries.append(ball)
                # combined_mesh += ball
                # pcd.normals = o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points)) * -1)
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                pcd.colors = o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points)) * colors[i])
                pcd.colors=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points))*colors[i])
                # o3d.visualization.draw_geometries([pcd])
                vis.add_geometry(pcd)
                combined_pcd += pcd
                obj_pcd += pcd

    for i,value in enumerate(bg_objects):
        if value is None or value['class_name'] == "ceiling" :
            continue
        pcd = selectPcdFromVoxel(value,0.25,shape,min_bound)
        pcd.colors=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points))*[211/255,211/255,211/255])
            # pcd = bg_objects[i]['pcd']
            # cl, ind = pcd.remove_radius_outlier(nb_points=8, radius=5)
            # pcd = pcd.select_by_index(ind)

        # pcd = selectPcdFromVoxel(bg_objects[i],voxel_size = 5, shape=(640,640,96))
        # pcd = bg_objects[i]['pcd']
        # cl, ind = pcd.remove_radius_outlier(nb_points=8, radius=5)
        # pcd = pcd.select_by_index(ind)
        # 删除 Z 坐标大于 50 的点
        points = np.asarray(pcd.points)
        # 获取点云的颜色数据
        colors = np.asarray(pcd.colors)
        # colors = filtered_colors
        index = np.where(points[:, 2] < 2.5)[0]
        if len(index) >0:
            points = points[index]
            colors = colors[index]
            # 将 RGB 转换成 HSL，调整亮度并转换回 RGB
            adjusted_colors = []
            for color in colors:
                # 1. 将 RGB 值（0-1）转换为 HSL
                r, g, b = color
                h, s, v = colorsys.rgb_to_hsv(r, g, b)

                # 2. 调低亮度，仅调整 l 参数
                v = max(0, v * 0.8)  # 确保亮度不会低于 0

                # 3. 将调整后的 HSL 转换回 RGB，不改变 H 和 S
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                adjusted_colors.append([r, g, b])
            pcd.points = o3d.utility.Vector3dVector(points)
            # 将调整后的颜色赋值回点云
            pcd.colors = o3d.utility.Vector3dVector(np.array(adjusted_colors))
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            vis.add_geometry(pcd)
            combined_pcd += pcd
            bg_pcd += pcd
        # scene_graph_geometries.append(ball)
    vis.capture_screen_image("connection.jpg")
    for id1, value in edges.items():
        for id2 in value.keys():
            if id2 in edges.keys():
                if int(id1) not in remove_list and int(id2) not in remove_list:
                    # print(id1,id2)
                    line_mesh = LineMesh(
                        points=np.array([objj_centers[int(id1)], objj_centers[int(id2)]]),
                        lines=np.array([[0, 1]]),
                        colors=[250 / 255, 109 / 255, 29 / 255],
                        radius=0.05
                    )
                    for cylinder_segment in line_mesh.cylinder_segments:
                        # 将每个圆柱体的顶点和面添加到combined_mesh
                        combined_mesh += cylinder_segment
                scene_graph_geometries.extend(line_mesh.cylinder_segments)

        for geometry in scene_graph_geometries:
            vis.add_geometry(geometry, reset_bounding_box=False)

    # vis.add_geometry(all_pcd)
    # vis.capture_screen_image("connection.jpg")
    render_option = vis.get_render_option()
    render_option.light_on = True
    # render_option.ambient_light = [0.5, 0.5, 0.5]

    if camera_pose is not None:
        # 提取视角信息
        view = camera_pose["trajectory"][0]
        front = np.array(view["front"])
        lookat = np.array(view["lookat"])
        up = np.array(view["up"])
        zoom = view["zoom"]
        boundingbox_max = np.array(view["boundingbox_max"])
        boundingbox_min = np.array(view["boundingbox_min"])
        # 创建AxisAlignedBoundingBox
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=boundingbox_min, max_bound=boundingbox_max)
        vis.add_geometry(bounding_box)

        # 设置摄像机视角
        ctr = vis.get_view_control()
        ctr.set_lookat(lookat)
        ctr.set_front(front)
        ctr.set_up(up)
        ctr.set_zoom(zoom)
    # vis.run()
    base_name = str(base_name)
    # 运行可视化
    vis.poll_events()  # 手动触发事件轮询
    vis.update_renderer()  # 更新渲染
    vis.capture_screen_image(f"{save_path}/{base_name}.png")

    # 延迟1秒后自动关闭窗口
    time.sleep(1)

    # 关闭窗口
    vis.close()

def draw_pcd_objects(objects,bg_objects,shape,min_bound,idx=None,pcd_path=None):
    pcd = o3d.geometry.PointCloud()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="test_poses_with_pcd")
    points=np.empty([0,3])
    normals=np.empty([0,3])
    colors=np.empty([0,3])
    for i, obj in enumerate(objects):
        if obj['exists_in_SG'] :
            coordinates = selectPointsFromVoxel(obj,0.25,shape,min_bound)
            index = np.where((coordinates[:, 2] < 2.5) & (coordinates[:, 2] > 0.08))[0]
            coordinates = coordinates[index]
            # print(idx,i,obj['class_name'],len(obj['voxel_index']),len(np.array(obj['pcd'].points)),len(coordinates))
            points = np.concatenate((points,coordinates),axis=0)
            normals = np.concatenate((normals,np.ones_like(coordinates)),axis=0)
            colors = np.concatenate((colors,np.ones_like(coordinates)*color_proposals[i%len(color_proposals)]),axis=0)
    
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # for obj in bg_objects.values() :
    #     if obj is not None :
    #         bg_pcd = selectPcdFromVoxel(obj,0.25,shape,min_bound)
    #         pcd = pcd + bg_pcd
    o3d.io.write_point_cloud(os.path.join(pcd_path,"obj/")+str(idx)+".ply", pcd)
    pcd = o3d.geometry.PointCloud()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="test_poses_with_pcd")
    points=np.empty([0,3])
    normals=np.empty([0,3])
    colors=np.empty([0,3])
    for i, value in enumerate(bg_objects):
        if value is not None :
            coordinates = selectPointsFromVoxel(value,0.25,shape,min_bound)
            index = np.where((coordinates[:, 2] < 2.5) )[0]
            coordinates = coordinates[index]
            # print(idx,i,obj['class_name'],len(obj['voxel_index']),len(np.array(obj['pcd'].points)),len(coordinates))
            points = np.concatenate((points,coordinates),axis=0)
            normals = np.concatenate((normals,np.ones_like(coordinates)),axis=0)
            colors = np.concatenate((colors,np.ones_like(coordinates)*color_proposals[i%len(color_proposals)]),axis=0)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # for obj in bg_objects.values() :
    #     if obj is not None :
    #         bg_pcd = selectPcdFromVoxel(obj,0.25,shape,min_bound)
    #         pcd = pcd + bg_pcd
    o3d.io.write_point_cloud(os.path.join(pcd_path,"bg/")+str(idx)+".ply", pcd)
def draw_pcd_with_objs(pose, pcd, img, objects):
    ln = len(pose)
    ans_pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="test_poses_with_pcd")
    for i, obj in enumerate(objects):
        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = obj['pcd'].points
        tmp_pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array(color_proposals[i]), (np.array(obj['pcd'].points).shape[0], 1)))
        # print('3333333')
        ans_pcd += tmp_pcd
    for i in range(ln):
        print('i:   {}'.format(i))
        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(pcd[i]['points'].reshape(-1, 3))
        tmp_pcd.colors = o3d.utility.Vector3dVector(img[i].cpu().numpy().reshape(-1, 3) / 255.)
        # tmp_pcd.transform(pose[i].cpu().numpy())
        ans_pcd += tmp_pcd

    vis.add_geometry(ans_pcd)
    vis.run()
    vis.destroy_window()
    print('draw_pcd')


if __name__ == '__main__':
    # path = '/home/caoyihan/Code/SGG/save_intermediate_element_for_SGG'
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='dataset path')
    args = parser.parse_args()
    begin, end = -1, 34
    pose, pcd, img = [], [], []
    for i in range(begin, end+1, 1):
        pkl_dict = read_pkl(os.path.join(args.path, 'save_dict_{}.pkl'.format(i)))
        pose.append(pkl_dict['pose'])
        pcd.append(pkl_dict['pcd_frame'])
        img.append(pkl_dict['rgb'])
        
    draw_pcd_with_objs(pose, pcd, img, pkl_dict['objects'])