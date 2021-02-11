# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%

import numpy as np
import math
import numpy
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as sp
#import pythreejs as threejs
#from pythreejs import *

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from selfsupmotion.data.objectron.dataset import iou
from selfsupmotion.data.objectron.dataset import box

import selfsupmotion.zero_shot_pose as zsp
# %%
import pandas as pd
from PIL import Image

# %%
experiment = "../output"

# %%
embeddings, info_df, train_embeddings, train_info_df = zsp.read_experiment(experiment)
train_info_df.head(2)

# %%
import random

#if ratio==1:
#    return info_df, embeddings
#ratio = 0.1
#info_df["video_uid"]=info_df["category"]+"_"+info_df["video_id"]
#video_uids = list(info_df["video_uid"].unique())
#len(video_uids)
#ratio = 0.1
#videos_uids_subset = random.sample(video_uids, int(len(video_uids)*ratio))
#print(f"Using a subset of {len(videos_uids_subset)} out of {len(video_uids)} total sequences for evaluation.")
#info_df_subset =train_info_df[train_info_df["video_uid"].isin(videos_uids_subset)]
#info_df_subset
#embeddings_subset=embeddings[:,list(info_df_subset.index)]
#embeddings_subset.shape
#info_df_subset = info_df_subset.reset_index()
#return info_df_subset, embeddings_subset

train_info_df_subset, train_embeddings_subset = zsp.get_subset(train_info_df, train_embeddings.get(), 0.1)
train_info_df_subset
# %%
sequence_annotations = zsp.get_sequence_annotations(info_df.iloc[1]["category"], info_df.iloc[1]["batch_number"], info_df.iloc[1]["sequence_number"])
sequence_annotations

# %%
zsp.get_points(info_df, 20)

# %%
camera = zsp.get_camera(info_df, 34000)
camera.intrinsics

# %%
train_info_df.head(2)


# %%
idx=100
Image.open(train_info_df.iloc[idx]["filepath_full"])
# %%
Image.open(train_info_df.iloc[idx]["filepath"])


# %%
idx=36000
points_2d, _ = zsp.get_points(info_df, idx)
im=Image.open(info_df.iloc[idx]["filepath_full"])
zsp.draw_bbox(im,zsp.points_2d_to_points2d_px(points_2d, im.width, im.height))
# %%
best_matches = np.argsort(-np.dot(embeddings[idx].T,train_embeddings)).get()
points_2d, _ = zsp.get_points(train_info_df, best_matches[0])
train_im=Image.open(train_info_df.iloc[best_matches[0]]["filepath_full"])

zsp.draw_bbox(train_im,zsp.points_2d_to_points2d_px(points_2d, im.width, im.height))
# %%
points2d_train, points3d_train = zsp.get_points(train_info_df, best_matches[0])
train_camera = zsp.get_camera(train_info_df, best_matches[0])
h_ratio = train_im.height / train_camera.image_resolution_height
w_ratio = train_im.width / train_camera.image_resolution_width

points2d_valid, points3d_valid = zsp.get_points(info_df, idx)
valid_image = Image.open(info_df.iloc[idx]["filepath_full"])
train_image = Image.open(train_info_df.iloc[best_matches[0]]["filepath_full"])
points2d_valid_px = zsp.points_2d_to_points2d_px(points2d_valid, valid_image.width, valid_image.height)
points2d_train_px = zsp.points_2d_to_points2d_px(points2d_train, train_image.width, train_image.height)

valid_bbox = zsp.get_bbox(points2d_valid_px, valid_image.width, valid_image.height)
train_bbox = zsp.get_bbox(points2d_train_px, train_image.width, train_image.height)
points2d_train_aligned = zsp.align_with_bbox(points2d_train_px, train_bbox, valid_bbox)
zsp.draw_bbox(valid_image, points2d_train_aligned)

# %%


points2d_valid, points3d_valid = zsp.get_points(info_df, idx)
#   points2d_valid_px = points_2d_to_points2d_px(points2d_valid, valid_image.width, valid_image.height)
#    points2d_train_px = points_2d_to_points2d_px(points2d_train, train_image.width, train_image.height)
valid_image = Image.open(info_df.iloc[idx]["filepath_full"])
train_image = Image.open(train_info_df.iloc[best_matches[0]]["filepath_full"])
    
points2d_train, points3d_train = zsp.get_points(train_info_df, best_matches[0])
points2d_valid_px = zsp.points_2d_to_points2d_px(points2d_valid, valid_image.width, valid_image.height)
points2d_train_px = zsp.points_2d_to_points2d_px(points2d_train, train_image.width, train_image.height)
valid_bbox = zsp.get_bbox(points2d_valid_px, valid_image.width, valid_image.height)
train_bbox = zsp.get_bbox(points2d_train_px, train_image.width, train_image.height)
points3d_train_rotated, points3d_train_aligned = zsp.align_with_bbox_3d(points3d_train, train_bbox, valid_bbox, verbose=True)


zsp.visualize(points3d_valid, points3d_train_rotated, points3d_train_aligned)
# %%
v_rotated = np.array(points3d_train_rotated)
v_aligned = np.array(points3d_train_aligned)
v_valid = np.array(points3d_valid)
w_rotated = box.Box(vertices=v_rotated)
w_aligned = box.Box(vertices=v_aligned)
w_valid = box.Box(vertices=v_valid)


loss = iou.IoU(w_rotated, w_valid)
loss_aligned = iou.IoU(w_aligned, w_valid)
print(f'iou rotated= {loss.iou()}, iou aligned={loss_aligned.iou()}')
print('iou (via sampling)= ', loss.iou_sampling())
# %%


zsp.get_iou(idx, embeddings, info_df, train_embeddings,
        train_info_df, k=0, show=True)

# %%
zsp.get_iou(17829, embeddings, info_df, train_embeddings,
        train_info_df, k=0, show=True)


# %%
import random
from tqdm import tqdm
subset = random.sample(range(0,len(info_df)),100)
ious = []
threshold = 0.5
valid_count = 0
for idx in tqdm(subset):
    iou_value, match_idx = zsp.get_iou(idx, embeddings, info_df, train_embeddings, train_info_df, ground_plane=False)
    if iou_value > threshold:
        valid_count+=1
    ious.append(iou_value)

ious = np.array(ious)
np.median(ious), ious.mean(), valid_count/len(subset)

assert ious.mean() > 0.25

# %%

sequence_annotations = zsp.get_sequence_annotations(train_info_df.iloc[idx]["category"], train_info_df.iloc[idx]["batch_number"], train_info_df.iloc[idx]["sequence_number"])
sequence_annotations.frame_annotations[0].camera.tracking_state
# %%
idx=38000
points2d, points3d = zsp.get_points(train_info_df, idx)
points2d
# %%
camera = zsp.get_camera(train_info_df, idx)
intrinsics = zsp.get_intrinsics(camera)
intrinsics

# %%
intrinsics[0,:]

# %%
scaled_intrinsics=zsp.scale_intrinsics(intrinsics, 0.25,0.25)
scaled_intrinsics
# %%
projected_points = zsp.project_3d_to_2d(points3d, scaled_intrinsics)
projected_points_h = np.hstack([projected_points, np.ones((len(projected_points),1))])
projected_points_h

# %%
zsp.points_2d_to_points2d_px(points2d, 360, 480)
# %%
zsp.describe_intrinsics(scaled_intrinsics)  
# %%
#1920x1440 -> 640x480. 0.25 scale factor

# %%
points3d_train = np.array(points3d_train)
res = np.dot(scaled_intrinsics, -points3d_train.T)
res/res[2]

#%%
zsp.points_2d_to_points2d_px(points2d_train, 480, 640)
# %%
np.dot(np.array(points2d), intrinsics)

# %%
np.linalg.inv(intrinsics)
# %%

# %%

# %%
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,9)
# %%
def plot_3d_box(points3d, reference_image, labels=None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    ax1.set_title("x-y / front view")
    ax1.set_title("image")
    ax3.set_title("z-x / top view")
    ax3.set_xlim([0,2])
    ax3.set_ylim([-1,1])
    ax4.set_xlim([0,2])
    ax4.set_ylim([-1,1])
    ax4.set_title("z-y / side view")
    if labels is None:
        labels = [str(i) for i in range(len(points3d))]
    
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax3.set_xlabel("z")
    ax3.set_ylabel("x")
    ax4.set_xlabel("z")
    ax4.set_ylabel("y")

    for point3d, label in zip(points3d, labels):
        y = -np.array(point3d)[:,0]
        x = np.array(point3d)[:,1]
        z = np.array(point3d)[:,2]
        ax1.scatter(x,y, label=label)
        ax2.imshow(reference_image)
        ax3.scatter(-z,x, label=label)
        ax4.scatter(-z,y, label=label)
    ax1.legend()
    ax3.legend()
    ax4.legend()
    
plot_3d_box([np.array(points3d_train),np.array(points3d_valid), np.array(points3d_train_rotated)], train_image, ["raw result","query","rotated result"])
# %%
plot_3d_box([np.array(points3d_train)], train_image, ["raw result"])

# %%
points_3d = []
labels = []
image_filepath_full=None
base=35600
for i in range(base,base+15):
    idx = i
    points2d, points3d = zsp.get_points(info_df, idx)
    points_3d.append(np.array(points3d))
    frame = info_df.iloc[idx]["frame"]
    labels.append(frame)
    image_filepath_full = frame = info_df.iloc[idx]["filepath_full"]

image = Image.open(image_filepath_full)
plot_3d_box(points_3d, image, labels)




# %%
#plane_center, plane_normal = zsp.get

# %%

# %%
match_idx = zsp.find_match_idx(idx, embeddings, train_embeddings)
# %%

# %%
plane_center, plane_normal = zsp.get_plane(info_df, idx)
points2d, points3d = zsp.get_points(info_df,idx)
match_idx  = zsp.find_match_idx(idx, embeddings, train_embeddings)
_, points3d_result = zsp.get_points(train_info_df,match_idx)
normal_tip = plane_center+ plane_normal/2

# %%
plot_3d_box([points3d_train, plane_center.reshape(1,3), normal_tip.reshape(1,3)], train_image, ["raw result", "plane center","plain normal tip"])

# %%

# %%
plane_center.reshape(1,3)
# %%

# %%

scene_plane = zsp.get_plane_equation_center_normal(plane_center, plane_normal)
scene_plane
# %%
# Unit test from: http://mathonline.wikidot.com/point-normal-form-of-a-plane#:~:text=Definition%3A%20Let%20be%20a%20plane,and%20is%20any%20point%20on%20.
zsp.get_plane_equation_center_normal(np.array([-2,3,4]),np.array([1,3,-7]))

# %%


plane1, plane2, plane3 = [1,0,0,-1], [0,1,0,-1],[0,0,1,-1]
zsp.get_planes_intersections(plane1, plane2, plane3)

# %% Bounding planes 
plane_rect = zsp.get_plane_points_z(scene_plane, plane_center, -1,-1,1,1)

# %%
match_idx = zsp.find_match_idx(idx, embeddings, train_embeddings,k=0)
zsp.get_match_aligned_points(idx, match_idx, info_df,  train_info_df)
# %%
camera_box = zsp.get_cube(1/20, center=(0,0,0))
# %%
import open3d as o3d
def visualize_point_cloud_in_bboxes(point_clouds, colors=None):
    """Small visualisation helper. Interface will change!

    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    if colors is None:
        colors = [[0,0,0] for x in range(len(point_clouds))]
    i=0
    for point_cloud, color in zip(point_clouds, colors):
        bbox_point_colors = zsp.get_objectron_bbox_colors()
        pcd = o3d.geometry.PointCloud()
        point_cloud = np.array(point_cloud)
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = bbox_point_colors
        bbox = zsp.create_objectron_bbox_from_points(point_cloud, color)
        print(i)
        i+=1
        vis.add_geometry(pcd)
        vis.add_geometry(bbox)
    
    vis.run()
    vis.destroy_window()
    return 


points3d = np.array(points3d)
zsp.get_middle_bottom_point(points3d, plane_normal)

# %%

0.5 * (points3d[zsp.BOTTOM_POINTS].min(axis=0) + points3d[zsp.BOTTOM_POINTS].max(axis=0))

# %%

zsp.get_3d_bbox_center(points3d)
# %%
points3d[0]



#

# %%
#idx = 36000 #Bon pour une démo
idx = 25600
plane_center, plane_normal = zsp.get_plane(info_df, idx)
points2d, points3d = zsp.get_points(info_df,idx)
scene_plane = zsp.get_plane_equation_center_normal(plane_center, plane_normal)
plane_rect = zsp.get_plane_points_z(scene_plane, plane_center, -1,-1,1,1)
camera_box = zsp.get_cube(1/20, center=(0,0,0))
camera_box[0]*=1.02
match_idx = zsp.find_match_idx(idx, embeddings, train_embeddings,k=0)
points3d_result_rotated, points3d_result =  zsp.get_match_aligned_points(idx, match_idx, info_df, train_info_df)
snapped, intersect = zsp.snap_box_to_plane(points3d_result_rotated, plane_normal, plane_center)
#snapped[0]=intersect
box_normal = snapped[0] - intersect
#box_normal[0]=-box_normal
box_rotation = zsp.rotation_matrix_from_vectors(box_normal, plane_normal)

FACES = np.array([
    [5, 6, 8, 7],  # +x on yz plane
    [1, 3, 4, 2],  # -x on yz plane
    [3, 7, 8, 4],  # +y on xz plane = top
    [1, 2, 6, 5],  # -y on xz plane = bottom
    [2, 4, 8, 6],  # +z on xy plane = front
    [1, 5, 7, 3],  # -z on xy plane
])
#points3d_result_rotated + 10*plane_normal

#%%
match_idx = zsp.find_match_idx(idx, embeddings, train_embeddings, k=0)
snapped, points3d_result = zsp.get_match_snapped_points(idx, match_idx, info_df, train_info_df)
# %%
pcd_snapped = o3d.geometry.PointCloud()
pcd_snapped.points = o3d.utility.Vector3dVector(np.array(snapped))
pcd_snapped.rotate(box_rotation, intersect)
# %%
zsp.get_3d_bbox_center(snapped)
# %%
snapped[0]

# %%
snapped_rotated = np.array(pcd_snapped.points)
snapped_rotated

# %%

normals = np.vstack([intersect, plane_normal+intersect, box_normal+intersect, snapped_rotated[0]])
normals

# %%
pcd_normals = o3d.geometry.PointCloud()
pcd_normals.points = o3d.utility.Vector3dVector(normals)

# %%
line_mapping = [(0,1), (0,2), (0,3)]
lineset_normals = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd_normals, pcd_normals, line_mapping)

bbox_plane_rect = zsp.create_objectron_bbox_from_points(plane_rect, (0,1,0))

# %%
camera = zsp.get_camera(info_df, idx)
intrinsics = zsp.get_intrinsics(camera)
intrinsics = zsp.scale_intrinsics(intrinsics, 0.25, 0.25)
points2d_px = zsp.project_3d_to_2d(snapped_rotated, intrinsics)
dest_bbox = zsp.get_bbox(points2d_px, 360, 480)

points2d_valid, points3d_valid = zsp.get_points(info_df, idx)
        
valid_image = Image.open(info_df.iloc[idx]["filepath_full"])
points2d_valid_px = zsp.points_2d_to_points2d_px(points2d_valid, valid_image.width, valid_image.height)
valid_bbox = zsp.get_bbox(points2d_valid_px, valid_image.width, valid_image.height)
scale_x, scale_y = zsp.get_scale_factors(dest_bbox, valid_bbox)
scale_factor = (scale_x + scale_y)/2


# %%
idx_max = int(np.argmax(np.linalg.norm(snapped_rotated,axis=1)))
#fixed_point = snapped_rotated[idx_max]
fixed_point = snapped_rotated[0]
snapped_rotated_scaled = snapped_rotated.copy()
#scale_factor = 1.2
snapped_rotated_scaled=(snapped_rotated_scaled-fixed_point)*scale_factor+fixed_point
snapped_rotated_scaled, intersect=zsp.snap_box_to_plane(snapped_rotated_scaled, plane_normal, plane_center)
snapped_rotated_scaled
# %%
vis = o3d.visualization.Visualizer()
vis.create_window()
pcd_snapped = o3d.geometry.PointCloud()
pcd_snapped.points = o3d.utility.Vector3dVector(snapped)

bbox_snapped = zsp.create_objectron_bbox_from_points(snapped, (1,0,0))
bbox_rotated_scaled = zsp.create_objectron_bbox_from_points(snapped_rotated_scaled,color=(0.5,0.5,0))
bbox_rotated = zsp.create_objectron_bbox_from_points(snapped_rotated, (0,1,0))
pcd_rotated = o3d.geometry.PointCloud()
pcd_rotated.points = o3d.utility.Vector3dVector(snapped_rotated)


bbox_query = zsp.create_objectron_bbox_from_points(np.array(points3d), (0,0,1))

#lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd, pcd, EDGES)
#mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.0003)
vis.add_geometry(pcd_snapped)
vis.add_geometry(pcd_normals)
#vis.add_geometry(bbox_snapped)
vis.add_geometry(bbox_rotated)
vis.add_geometry(lineset_normals)
vis.add_geometry(bbox_query)
vis.add_geometry(bbox_plane_rect)
vis.add_geometry(bbox_rotated_scaled)
#vis.add_geometry(lineset)

vis.run()
vis.destroy_window()

# %%
visualize_point_cloud_in_bboxes(
    [plane_rect, points3d, camera_box,points3d_result+plane_normal, points3d_result_rotated, snapped, snapped_rotated, snapped_rotated_scaled],
    [[0,0,0],          [1,0,0],[1,1,0],     [0,0,1],             [0,0.5,0], [0,0.75,0],  [0,1,0], [0.5,0.5,0]])



# %%
samples = random.sample(list(range(0,len(info_df))),k=100)


# %%
idx = 1673
match_idx = zsp.find_match_idx(idx, embeddings, train_embeddings)
snapped, points3d_result = zsp.get_match_snapped_points(idx, match_idx, info_df, train_info_df)
plane_center, plane_normal = zsp.get_plane(info_df, idx)
face_facing_plane = zsp.find_face_facing_plane_v2(snapped, plane_normal)
face_facing_plane
# %%
points = points3d_result_rotated

    #closest_points = np.argsort(dist[1:9])+1
    #closest_points
    #for face in FACES:
    #    if sorted(face)==sorted(closest_points[0:4]):
    #        return face
    #raise ValueError("Ambiguous face!")


# %%
error = 0
for idx in samples:
    match_idx = zsp.find_match_idx(idx, embeddings, train_embeddings)
    snapped, points3d_result = zsp.get_match_snapped_points(idx, match_idx, info_df, train_info_df)
    plane_center, plane_normal = zsp.get_plane(info_df, idx)
    try:
        face_facing_plane = zsp.find_face_facing_plane_v2(snapped, plane_normal)
    except ValueError:
        face_facing_plane = [1, 2, 6, 5]
        print(idx)
#print(error)
# %%
point = zsp.get_middle_bottom_point(points3d_result_rotated, plane_normal)

# %%
snapped_rotated
line_mapping = [(1,3)]
lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd_rotated, pcd_rotated, line_mapping)

# %%
np.linalg.norm(plane_normal)

# %%
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(lineset)
vis.add_geometry(pcd_rotated)
vis.run()
vis.destroy_window()
# %%
axis = snapped_rotated[3]-snapped_rotated[1]
angle = np.array([(5/180)*np.pi,0,0])
axis, angle
# %%
#mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
#mesh.get_rotation_matrix_from_axis_angle(axis, angle)

theta = (10/180)*np.pi
u = snapped_rotated[3]-snapped_rotated[1]
R = zsp.get_rotation_matrix_vector_angle(u, theta)
R
# %%
np.cos(theta) + (u[0]**2)*(1-np.cos(theta))
# %%
pcd_rotated1 = o3d.geometry.PointCloud(pcd_rotated)
pcd_rotated1.rotate(R,center=snapped_rotated[0])

# %%
def rotate_around_center(points, R, center):
    center = points[0]
    original_centered = points-center
    original_centered_rotated = np.dot(R, original_centered.T).T
    rotated = original_centered_rotated + center
    return rotated
rotate_around_center(snapped_rotated, R, snapped_rotated[0])

def rotate_bbox_around_its_center(points, theta):
    center = points[0]
    axis = points[3]-points[1]
    R = zsp.get_rotation_matrix_vector_angle(axis, theta)
    return rotate_around_center(points, R, center)

# %%
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(lineset)
vis.add_geometry(pcd_rotated)
#vis.add_geometry(pcd_rotated1)
u = snapped_rotated[3]-snapped_rotated[1]
for angle in np.arange(5,360,5):
    theta = (angle/180)*np.pi
    pivoted = rotate_bbox_around_its_center(snapped_rotated, theta)
    #R = zsp.get_rotation_matrix_vector_angle(u, theta)
    #pivoted = rotate_around_center(snapped_rotated, R, snapped_rotated[0])
    #pcd_rotated1 = o3d.geometry.PointCloud(pcd_rotated)
    #pcd_rotated1.rotate(R,center=snapped_rotated[0])
    vis.add_geometry(zsp.create_objectron_bbox_from_points(pivoted))
vis.add_geometry(zsp.create_objectron_bbox_from_points(camera_box))
vis.add_geometry(zsp.create_objectron_bbox_from_points(plane_rect))
vis.run()
vis.destroy_window()


# %%
info_df.query("category=='book'")

# %%
