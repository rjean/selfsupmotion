#from notebooks.opencv_install.opencv_contrib.modules.dnn_objdetect.scripts.pascal_preprocess import rescale
import PIL
import numpy
from numpy.core.fromnumeric import argmax
import pandas as pd
import open3d as o3d
import random
import math
#import multiprocessing
from multiprocessing import Pool
#multiprocessing.set_start_method('spawn')
import numpy as np
import cupy as cp
import os

from tqdm import tqdm

from deco import concurrent, synchronized
from tqdm.utils import RE_ANSI

import selfsupmotion.data.objectron.data_transforms
import selfsupmotion.data.objectron.sequence_parser
import selfsupmotion.data.utils

from selfsupmotion.data.objectron.dataset import iou
from selfsupmotion.data.objectron.dataset import box

from google.protobuf import text_format
# The annotations are stored in protocol buffer format. 
from selfsupmotion.data.objectron.schema import annotation_data_pb2 as annotation_protocol
# The AR Metadata captured with each frame in the video
from selfsupmotion.data.objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol

from PIL import Image, ImageDraw
EDGES = (
    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
)

FACES = np.array([
    [5, 6, 8, 7],  # +x on yz plane
    [1, 3, 4, 2],  # -x on yz plane
    [3, 7, 8, 4],  # +y on xz plane = top
    [1, 2, 6, 5],  # -y on xz plane = bottom
    [2, 4, 8, 6],  # +z on xy plane = front
    [1, 5, 7, 3],  # -z on xy plane
])

BOTTOM_POINTS = [1, 2, 6, 5]

use_cupy = True

def get_center_ajust(result_bbox, projected_center):
    cx_result = (result_bbox[0] + result_bbox[2])/2
    cy_result = (result_bbox[1] + result_bbox[3])/2
    adjust_x = projected_center[0] - cx_result
    adjust_y = projected_center[1] - cy_result
    adjust_x_rel = adjust_x / (result_bbox[2]-result_bbox[0])
    adjust_y_rel = adjust_y / (result_bbox[3]-result_bbox[1])
    return adjust_x_rel, adjust_y_rel

def estimate_object_center_in_query_image(query_intrinsics, query_bbox, points_2d_px_result):
    cx = (query_bbox[0] + query_bbox[2])/2
    cy = (query_bbox[1] + query_bbox[3])/2
    result_bbox = get_bbox(points_2d_px_result, 360, 480) 
    projected_center = points_2d_px_result[0]
    adjust_x_rel, adjust_y_rel = get_center_ajust(result_bbox, projected_center)
    adjust_x_px = adjust_x_rel * (query_bbox[2]-query_bbox[0])
    adjust_y_px = adjust_y_rel * (query_bbox[3]-query_bbox[1])
    cx = cx + adjust_x_px
    cy = cy + adjust_y_px
    b = (cx - query_intrinsics[1,2])/query_intrinsics[1,1]
    a = (cy - query_intrinsics[0,2])/query_intrinsics[0,0]
    return cx, cy, a, b

def get_dist_from_plane(plane_normal, plane_center, point):
    plane = get_plane_equation_center_normal(plane_center, plane_normal)
    a, b, c, d = plane
    X,Y,Z = point
    return abs(a*X + b*Y + c*Z + d)/np.sqrt(a**2+b**2+c**2)

def intersect_plane_with_line(plane_normal, plane_center, point1, point0=np.array([0,0,0])):
    p0 = plane_center
    n = plane_normal
    l0 = point0 #Line start.
    l = point1 #Line end
    d = np.dot((p0-l0), n)/np.dot(l,n)
    intersect = l0 + d*l
    return intersect

def get_bbox_area(bbox):
    dx = bbox[2]-bbox[0]
    dy = bbox[3]-bbox[1]
    return np.sqrt(dx**2+dy**2)

def get_scale_factor(points_3d_query, points3d_scaled, intrinsics, width=360, height=480):
    points2d_px_result = project_3d_to_2d(points3d_scaled, intrinsics)
    points2d_px_query = project_3d_to_2d(points_3d_query, intrinsics)
    result_bbox = get_bbox(points2d_px_result, width, height)
    query_bbox = get_bbox(points2d_px_query, width, height)
    scale = get_bbox_area(query_bbox) / get_bbox_area(result_bbox)
    return scale

def align_box_with_plane(points3d, plane_normal_query, plane_normal_result):
    box_rotation = rotation_matrix_from_vectors(plane_normal_query, plane_normal_result)
    #plane_normal_query, plane_normal_result, plane_center_query, plane_center_result
    points_3d_axis = np.dot(points3d-points3d[0],box_rotation)+points3d[0]
    return points_3d_axis

def get_smooth_scale_factor(points_3d_query, points3d_scaled, intrinsics, alpha):
    factor = get_scale_factor(points_3d_query, points3d_scaled, intrinsics)    
    return (alpha+factor-1)/alpha

def get_bounding_box(idx, match_idx, info_df, train_info_df, adjust_scale=False):
    points_2d_result, points_3d_result = get_points(train_info_df, match_idx)
    points_2d_px_result = points_2d_to_points2d_px(points_2d_result, 360, 480)
    points_2d_query, points_3d_query = get_points(info_df, idx)
    points_2d_px_query = points_2d_to_points2d_px(points_2d_query, 360, 480)
    plane_center_query, plane_normal_query= get_plane(info_df, idx)
    plane_center_result, plane_normal_result = get_plane(train_info_df, match_idx)
    result_bbox = get_bbox(points_2d_px_result, 360, 480) 
    query_camera = get_camera(info_df, idx)
    query_intrinsics = get_intrinsics(query_camera)
    query_intrinsics = scale_intrinsics(query_intrinsics, 0.25,0.25)
    result_bbox = get_bbox(points_2d_px_result, 360, 480)
    query_bbox = get_bbox(points_2d_px_query, 360, 480)
    cx, cy, a, b = estimate_object_center_in_query_image(query_intrinsics, query_bbox, points_2d_px_result)
    center_ray = np.array([a,b,-1])
    points_3d_axis = align_box_with_plane(points_3d_result, plane_normal_query, plane_normal_result)
    obj_radius = get_dist_from_plane(plane_normal_result, plane_center_result, points_3d_axis[0])
    points_3d_result_snapped = snap_to_plane(points_3d_axis, plane_normal_query, plane_center_query, center_ray, obj_radius)
    if adjust_scale:
        scale = get_smooth_scale_factor(points_3d_query, points_3d_result_snapped, query_intrinsics, 2)
        points3d_scaled = points_3d_result_snapped
        for i in range(0,4):
            #print(scale, obj_radius)
            obj_radius = get_dist_from_plane(plane_normal_query, plane_center_query, points3d_scaled[0])
            points3d_scaled = snap_to_plane(scale_3d_bbox(points3d_scaled, scale),
                                                plane_normal_query, plane_center_query, center_ray, obj_radius = obj_radius*scale)
            scale = get_smooth_scale_factor(points_3d_query, points3d_scaled, query_intrinsics, 2)
            #print(i, get_iou_between_bbox(np.array(points_3d_query), np.array(points3d_scaled)))
        return points3d_scaled
    return points_3d_result_snapped

def scale_3d_bbox(bbox, factor):
    assert factor>0
    center = bbox[0]
    bbox = (bbox-center)*factor + center 
    return bbox

def snap_to_plane(points3d, plane_normal_query, plane_center_query, point_on_center_line, obj_radius=None):
    #if obj_radius is None:
    #    obj_radius = get_dist_from_plane(plane_normal_result, plane_center_result, points3d[0])
    offset = obj_radius * plane_normal_query
    new_center = intersect_plane_with_line(plane_normal_query, plane_center_query+offset, point_on_center_line) 
    snapped = points3d - points3d[0] + new_center
    return snapped

def create_objectron_bbox_from_points(points, color=(0,0,0)):
    assert len(color)==3
    #EDGES = (
    #    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    #    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    #    [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
    #)
    pcd = o3d.geometry.PointCloud()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd, pcd, EDGES)
    colors = np.array(color*len(lineset.lines)).reshape(len(lineset.lines),3)
    lineset.colors=o3d.utility.Vector3dVector(colors)
    return lineset

# %%
# From Stack Overflow: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -float(v[2]), float(v[1])], [float(v[2]), 0, -float(v[0])], [-float(v[1]), float(v[0]), 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


# %%
#https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

def get_rotation_matrix_vector_angle(u, theta):
    R = np.zeros((3,3))
    ux, uy, uz = u/np.linalg.norm(u)
    #Diagonal
    R[0,0] = np.cos(theta) + (ux**2)*(1-np.cos(theta))
    R[1,1] = np.cos(theta) + (uy**2)*(1-np.cos(theta))
    R[2,2] = np.cos(theta) + (uz**2)*(1-np.cos(theta))
    # Row 1
    R[0,1] = ux*uy*(1-np.cos(theta)) - uz*np.sin(theta)
    R[0,2] = ux*uz*(1-np.cos(theta)) + uy*np.sin(theta)
    #Row 2
    R[1,0] = uy*ux*(1-np.cos(theta)) + uz*np.sin(theta)
    R[1,2] = uy*uz*(1-np.cos(theta)) - ux*np.sin(theta)
    #Row 3
    R[2,0] = uz*ux*(1-np.cos(theta)) - uy*np.sin(theta)
    R[2,1] = uz*uy*(1-np.cos(theta)) + ux*np.sin(theta)
    return R

def rotate_around_center(points, R, center=(0,0,0)):
    center = points[0]
    original_centered = points-center
    original_centered_rotated = np.dot(R, original_centered.T).T
    rotated = original_centered_rotated + center
    return rotated

def rotate_bbox_around_its_center(points, theta):
    center = points[0]
    axis = points[3]-points[1]
    R = get_rotation_matrix_vector_angle(axis, theta)
    return rotate_around_center(points, R, center)

BOTTOM_POINTS = [1, 2, 6, 5]
def get_middle_bottom_point(points3d, plane_normal):
    face = find_face_facing_plane_v2(points3d,plane_normal)
    return points3d[face].mean(axis=0)
    #return points3d[BOTTOM_POINTS].mean(axis=0)

def snap_box_to_plane(points3d, plane_normal, plane_center, align_axis=True):
    bottom_middle=get_middle_bottom_point(points3d, plane_normal)
    intersect = get_intersect_relative_to_camera(plane_normal, plane_center, bottom_middle)
    snapped= points3d-(bottom_middle-intersect)
    if align_axis:
        box_normal = snapped[0] - intersect
        #box_normal[0]=-box_normal
        box_rotation = rotation_matrix_from_vectors(box_normal, plane_normal)
        pcd_snapped = o3d.geometry.PointCloud()
        pcd_snapped.points = o3d.utility.Vector3dVector(np.array(snapped))
        pcd_snapped.rotate(box_rotation, intersect)
        snapped_rotated = np.array(pcd_snapped.points)
        snapped = snapped_rotated
    return snapped, intersect

def get_cube(scaling_factor=1, center=(0,0,0)):
    cube= (
        (0, 0, 0 ),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, -1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, -1, 1),
        (-1, 1, 1)
        )
    cube = np.array(cube)*scaling_factor
    cube += np.array(center)
    return cube

def get_match_aligned_points(idx:int, match_idx:int, info_df: pd.DataFrame, 
            train_info_df: pd.DataFrame,
            ground_truth=False):
    """Get 3D IoU for a specific query image, using the kth neighbor.

    Returns:
        [type]: [description]
    """
    assert type(match_idx)==int
    points2d_train, points3d_train = get_points(train_info_df, match_idx)
    points2d_valid, points3d_valid = get_points(info_df, idx)
        
    valid_image = Image.open(info_df.iloc[idx]["filepath_full"])
    train_image = Image.open(train_info_df.iloc[match_idx]["filepath_full"])
    points2d_valid_px = points_2d_to_points2d_px(points2d_valid, valid_image.width, valid_image.height)
    points2d_train_px = points_2d_to_points2d_px(points2d_train, train_image.width, train_image.height)
    valid_bbox = get_bbox(points2d_valid_px, valid_image.width, valid_image.height)
    train_bbox = get_bbox(points2d_train_px, train_image.width, train_image.height)
    points3d_train_rotated, _ = align_with_bbox_3d(points3d_train, train_bbox, valid_bbox)
    if ground_truth:
        return np.array(points3d_train_rotated), np.array(points3d_valid)
    else:
        return np.array(points3d_train_rotated), np.array(points3d_train)


def get_plane_points_z(scene_plane: np.ndarray, plane_center: np.ndarray, xmin=-3, ymin=-3, xmax=3, ymax=3):
    """Create a bounding box representing the ground plane around the origin for visualisation.

    Args:
        scene_plane (np.ndarray): Scene plane equation [ax by cz d]
        plane_center (np.ndarray): Plane center of the scene.
        xmin (int, optional): [description]. Defaults to -3.
        ymin (int, optional): [description]. Defaults to -3.
        xmax (int, optional): [description]. Defaults to 3.
        ymax (int, optional): [description]. Defaults to 3.

    Returns:
        np.array: Ground plane bounding box.
    """
    bottom_plane = [0,1,0,-ymin] #y=-1
    top_plane = [0,1,0,-ymax] #y=1
    left_plane = [1,0,0,-xmin] #x=-1
    right_plane = [1,0,0,-xmax] #x=1
    top_left = get_planes_intersections(top_plane, left_plane, scene_plane)
    bottom_left = get_planes_intersections(bottom_plane, left_plane, scene_plane)
    top_right = get_planes_intersections(top_plane, right_plane, scene_plane)
    bottom_right = get_planes_intersections(bottom_plane, right_plane, scene_plane)

    eps=0.01
    #EDGES = (
    #[1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    #[1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    #[1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
    #)
    plane_points = np.vstack([top_left, bottom_left, top_right, bottom_right])
    plane_points_offset = np.vstack([top_left-eps, bottom_left-eps, top_right-eps, bottom_right-eps])
    plane_rect = np.vstack((plane_points, plane_points_offset, plane_center))
    return plane_rect

def get_plane_equation_center_normal(plane_center, plane_normal):
    #ax +by + cz = d 
    #where d= ax0 + by0 +czo
    #https://tutorial.math.lamar.edu/classes/calciii/eqnsofplanes.aspx

    #The plane equation is [ax + by +cz -d]=0
    return np.hstack((plane_normal, -np.dot(plane_center,plane_normal)))

def get_planes_intersections(plane1, plane2, plane3):
    #See Hartley, p.589
    #The plane equation is [ax + by +cz -d]=0
    A = np.vstack((plane1, plane2, plane3))
    u,d,v= np.linalg.svd(A)
    v = v #Numpy returns V transposed. We need v.
    u,d,v
    point = v[-1]
    point = point / point[-1] #Normalization of homogenous coordinates.
    return point[0:3]

#https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
def get_intersect_relative_to_camera(plane_normal, plane_center, point):
    p0 = plane_center
    n = plane_normal
    l0 = np.array([0,0,0]) #camera center
    l = point #Relative to camera center
    d = np.dot((p0-l0), n)/np.dot(l,n)
    intersect = l0 + d*l
    return intersect

def get_3d_bbox_center(points3d):
    return 0.5 * (points3d[1:].min(axis=0) + points3d[1:].max(axis=0))

def get_objectron_bbox_colors():
    bbox_point_colors = np.zeros((9,3))
    
    for bottom_point in BOTTOM_POINTS:
        bbox_point_colors[bottom_point]=np.array([1,0,0])
    bbox_point_colors[0] = np.array([0,0,1])
    return o3d.utility.Vector3dVector(bbox_point_colors)  

def find_face_facing_plane_v2(points, plane_normal):
    min_dist=1e6
    most_probable_face = [1, 2, 6, 5]
    for face in FACES:
        points_from_current_face = points[face]
        closest_point_idx = int(np.argmin(np.dot(plane_normal, (points_from_current_face+plane_normal*20).T)))
        closest_corner = points_from_current_face[closest_point_idx]
        points_relative_to_closest_corner = points_from_current_face-closest_corner
        normalized = points_relative_to_closest_corner/np.linalg.norm(points_relative_to_closest_corner, axis=1).reshape(4,1)
        normalized[closest_point_idx] = np.array([0,0,0])
        dist = np.dot(plane_normal, (closest_corner+normalized+plane_normal*20).T).mean()
        if dist<min_dist:
            most_probable_face=face
            min_dist = dist
    return most_probable_face

def get_plane(df : pd.DataFrame, idx: int):
    """Get object plane for a specific camera frame.

    Args:
        df (pd.DataFrame): Embeddings meta data frame.
        idx (int): Index of the frame in the dataframe.

    Returns:
        tuple: 2d points, 3d points relative to camera.
    """
    idx = int(idx)
    sequence_annotations = get_sequence_annotations(df.iloc[idx]["category"], df.iloc[idx]["batch_number"], df.iloc[idx]["sequence_number"])
    frame = int(df.iloc[idx]["frame"])

    plane_center = sequence_annotations.frame_annotations[frame].plane_center
    plane_normal = sequence_annotations.frame_annotations[frame].plane_normal
    
    return np.array(plane_center), np.array(plane_normal)

def parse_info_df(info_df, subset="valid"):
    """Parses an embedding meta-data dataframe. (The output of SimSiam)

    Args:
        info_df (pd.DataFrame): Pandas Dataframe, as output by the "read experiment" function.
        subset (str, optional): Which subset to use. Can be "train", "valid" or test. Defaults to "valid".
    """
    info_df["category"]=info_df["uid"].str.extract("(.*?)-")
    info_df["sequence_uid"]=info_df["uid"].str.extract("(batch\-\d+_\d+_\d+)")
    info_df["frame"]=info_df["uid"].str.extract("-(\d+)$")
    info_df["video_id"]=info_df["uid"].str.extract("(batch\-\d+_\d+)")
    info_df["object_id"]=info_df["uid"].str.extract("batch\-\d+_\d+_(\d+)")
    info_df["batch_number"]=info_df["uid"].str.extract("(batch\-\d+)")
    info_df["sequence_number"]=info_df["uid"].str.extract("batch\-\d+_(\d+)_\d+")
    info_df["filepath"]=f"/home/raphael/datasets/objectron/96x96/{subset}/" + info_df["category"] +"/" + info_df["sequence_uid"] +"." + info_df["frame"] + ".jpg"
    info_df["filepath_full"]="/home/raphael/datasets/objectron/640x480_full/" + info_df["category"] +"/" + info_df["sequence_uid"] +"." + info_df["frame"] + ".jpg"

def read_experiment(experiment: str):
    """Read the output of a SimSiam experiment folder. (Embeddings and metadata)

    Args:
        experiment (str): Output folder location 

    Returns:
        tuple: validation embeddings, validation meta data, train embeddings, train meta data
    """
    global use_cupy
    load_fn = cp.load
    if not use_cupy:
        load_fn = np.load
    
    embeddings = load_fn(f"{experiment}/embeddings.npy")
    if embeddings.shape [1]>embeddings.shape [0]:
        embeddings=embeddings.T
    info = numpy.load(f"{experiment}/info.npy")
    assert info.shape[0]==2
    train_new_filename = f"{experiment}/train_embeddings.npy"
    train_old_filename = f"{experiment}/training_embeddings.npy"
    if os.path.exists(train_new_filename):
        train_embeddings = load_fn(train_new_filename)
    else:
        train_embeddings = load_fn(train_old_filename)
    train_info = numpy.load(f"{experiment}/train_info.npy")
    assert train_info.shape[0]==2
    info_df = pd.DataFrame(info.T)
    train_info_df = pd.DataFrame(train_info.T)
    info_df_columns = {0:"category_id",1:"uid"}
    train_info_df = train_info_df.rename(columns=info_df_columns)
    info_df = info_df.rename(columns=info_df_columns)
    parse_info_df(info_df)
    parse_info_df(train_info_df, subset="train")
    assert train_embeddings.shape[0] == embeddings.shape[1]
    return embeddings, info_df, train_embeddings, train_info_df


def get_sequence_annotations(category: str, batch_number: str, sequence_number: str, annotations_path="/home/raphael/datasets/objectron/annotations"):
    """Get annotation data for a video sequence.

    Args:
        category (str): Category in the objectron dataset.
        batch_number (str): Batch number. 
        sequence_number (str): Sequence number. 
        annotations_path (str, optional): Location of the Objectron annotation files. Defaults to "/home/raphael/datasets/objectron/annotations".

    Returns:
        annotations: Google's annotations in their format.
    """
    sequence = annotation_protocol.Sequence()
    #category, batch_number, sequence_number = get_batch_sequence_from_video_file(video_file)
    annotation_file=f"{annotations_path}/{category}/{batch_number}/{sequence_number}.pbdata"
    with open(annotation_file, 'rb') as pb:
        sequence.ParseFromString(pb.read())
    return sequence


def get_points(df : pd.DataFrame, idx: int):
    """Get 2d and 3d points for a specific frame.

    Args:
        df (pd.DataFrame): Embeddings meta data frame.
        idx (int): Index of the frame in the dataframe.

    Returns:
        tuple: 2d points, 3d points relative to camera.
    """
    idx = fix_idx(idx)
    sequence_annotations = get_sequence_annotations(df.iloc[idx]["category"], df.iloc[idx]["batch_number"], df.iloc[idx]["sequence_number"])
    frame = int(df.iloc[idx]["frame"])
    object_id = int(df.iloc[idx]["object_id"])
    #print(frame)
    #sequence_annotations.frame_annotations[0].annotations[0]
    keypoints = sequence_annotations.frame_annotations[frame].annotations[object_id].keypoints
    points_2d = []
    points_3d = []
    for keypoint in keypoints:
        point_2d = (keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth)
        point_3d = (keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z)
        points_2d.append(point_2d)
        points_3d.append(point_3d)
    return np.array(points_2d), np.array(points_3d)


def fix_idx(idx):
    if type(idx) is not int and type(idx) is not numpy.ndarray:
        idx = int(idx) #Just make sure this is the right 
    return idx

def get_camera(df: pd.DataFrame, idx:int):
    """Get camera information for a specific trame. 

    Args:
        df (pd.DataFrame): Embeddings meta data frame.
        idx (int): Index of the frame in the dataframe.

    Returns:
        object: Google's camera information for the frame.
    """
    idx = fix_idx(idx)
    sequence_annotations = get_sequence_annotations(df.iloc[idx]["category"], df.iloc[idx]["batch_number"], df.iloc[idx]["sequence_number"])
    frame = int(df.iloc[idx]["frame"])
    frame_annotation = sequence_annotations.frame_annotations[frame]
    return frame_annotation.camera


def get_bbox(points2d_px: list, width: int, height: int, clip=True):
    """Get 2d bounding box in pixel for a normalized 2d point list.

    Args:
        points2d_px (list): List of normalized 2d points.
        width (int): Image width in pixels.
        height (int): Image heigh in pixels.
        clip (bool, optional): Clip values outside of picture. Defaults to True.

    Returns:
        tuple: x_min, y_min, x_max, y_max in pixels.
    """
    x_min = 10000
    x_max = 0
    y_min = 10000
    y_max = 0
    for point2d_px in points2d_px:
        x,y = point2d_px
        #x*=width
        #y*=height
        if x < x_min:
            x_min=x
        if x > x_max:
            x_max=x
        if y < y_min:
            y_min=y
        if y > y_max:
            y_max=y
    if clip:
        x_min=max(x_min,0)
        y_min=max(y_min,0)
        x_max=min(x_max,width)
        y_max=min(y_max,height)
    return x_min, y_min, x_max, y_max


def get_center(x_min: int,y_min: int,x_max:int,y_max:int):
    """Get center of bounding box.

    Args:
        x_min (int): Minimum x value in pixels.
        y_min (int): Minimum y value in pixels.
        x_max (int): Maximum x value in pixels.
        y_max (int): Maximum y value in pixels.

    Returns:
        tuple: Center (x,y) in pixels.
    """
    x = (x_min+x_max)/2
    y = (y_min+y_max)/2
    return x,y

def points_2d_to_points2d_px(points2d:list, width:int, height:int):
    """Convert normalzied 2d point list into pixel point list.

    Args:
        points2d (list): Normalized 2d points list (x, y, depth)
        width (int): Image width in pixels.
        height (int): Image heigh in pixels.

    Returns:
        list: (x,y) in pixels relative to image. 
    """
    points2d_px=[]
    for i, point2d in enumerate(points2d):
        x = point2d[0]*width
        y = point2d[1]*height
        points2d_px.append((x,y))
    return np.array(points2d_px)

def get_scale_factors(bbox1:tuple, bbox2:tuple):
    """Get x and y scaling factors between 2 bounding boxes.

    Args:
        bbox1 (tuple): xmin1, ymin1, xmax1, ymax1
        bbox2 (tuple): xmin2, ymin2, xmax2, ymax2

    Returns:
        tuple: (x,y) scale factor.
    """
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    width1 = xmax1-xmin1
    width2 = xmax2-xmin2
    height1 = ymax1-ymin1
    height2 = ymax2-ymin2
    try:
        return float(width2/width1), float(height2/height1)
    except ZeroDivisionError:
        return (1,1)

def align_with_bbox(train_point2d_px, train_bbox, valid_bbox):
    """Aligns a projected 3d bounding box in pixel space.

    Args:
        train_point2d_px (list): Projected points in pixels of a 3d bounding box relative to image.
        train_bbox (tuple): Result 2d bounding box, in pixels.
        valid_bbox ([type]): Query 2d bounding box, in pixels.
    """
    x_center_valid, y_center_valid = get_center(*valid_bbox)
    x_center_train, y_center_train = get_center(*train_bbox)
    
    #x_center_train, y_center_train = train_point2d_px[0][0], train_point2d_px[0][1]
    dx = x_center_valid - x_center_train
    dy = y_center_valid - y_center_train
    aligned_points2d_px = []
    scale_x, scale_y = get_scale_factors(train_bbox, valid_bbox)
#    scale_x = 1.2
#    scale_y = 1.2
    for train_point2d_px in train_point2d_px:
        x,y = train_point2d_px
        x+=dx
        y+=dy
        x = (x-x_center_train)*scale_x + x_center_train
        y = (y-y_center_train)*scale_y + y_center_train
        aligned_points2d_px.append((x,y))
    
    return aligned_points2d_px

def draw_bbox(im: PIL.Image, points2d_px: list, line_color=(0,255,0), pixel_center_color=(0,255,0), object_center_color=(255,0,0)):
    """Draw a projected 2d bounding box (in pixels) over a Pillow Image

    Args:
        im (PIL.Image): Query image.
        points2d_px (list): Aligned projected 2d bounding box in pixels.

    Returns:
        Pil.Image: Image with drawn bounding box.
    """
    #with Image.open(image_file) as im:
    #points2d, points3d = 
    draw = ImageDraw.Draw(im)
    #points2d_px=[]
    #points2d_px = points_2d_to_points2d_px(points2d, im.width, im.height)
    for i, point2d_px in enumerate(points2d_px):
        x, y = point2d_px
        fill=(0,0,255)
        if i==0: #Center point
            fill=object_center_color
        draw.ellipse((x-5 , y-5,x+5 , y+5), fill=fill)
    #points2d_px.append((x,y))
    
    x_min, y_min, x_max, y_max = get_bbox(points2d_px, im.width, im.height)
    x_center = (x_max+x_min)/2
    y_center = (y_max+y_min)/2
    draw.ellipse((x_center-10 , y_center-10,x_center+10 , y_center+10), fill=pixel_center_color)
    for edge in EDGES:
        p1, p2 = edge
        x1, y1 = points2d_px[p1]
        x2, y2 = points2d_px[p2]
        draw.line((x1,y1,x2,y2), fill=line_color)
    return im
import math 

def align_with_bbox_3d(points3d_train: list, train_bbox: tuple, valid_bbox: tuple, 
                        alpha_x=368.0, alpha_y:float=None, verbose=False):
    """Aligns a 3d bounding box (result) according to the query bounding box location and the
    result bounding box in 2d (pixels).

    Args:
        points3d_train (list): 3d bounding box relative to camera. First point is center.
        train_bbox (tuple): Result 2d bounding box location in pixels.
        valid_bbox (tuple): Query 2d bounding box location in pixels.
        alpha_x (float, optional): Alpha x intrinsic camera parameter. Defaults to 368.0.
        alpha_y ([type], optional): Alpha y intrinsic camera parameter. Defaults to None:float.
        verbose (bool, optional): [description]. Defaults to False.
    """
    
    if alpha_y is None:
        alpha_y = alpha_x #Square CCD
    x_center_valid, y_center_valid = get_center(*valid_bbox)
    x_center_train, y_center_train = get_center(*train_bbox)
    
    #x_center_train, y_center_train = train_point2d_px[0][0], train_point2d_px[0][1]
    dx_px = x_center_valid - x_center_train
    dy_px = y_center_valid - y_center_train
    
    translated_points3d = []
    scale_x, scale_y = get_scale_factors(train_bbox, valid_bbox)

    scale = 1-(scale_x+scale_y)/2
    distance_scale = 1/((scale_x+scale_y)/2)
    c_x, c_y, c_z = points3d_train[0]
    c_depth = np.sqrt(c_x**2+c_y**2+c_z**2) 
    #print(dx_px, dy_px, c_z)
    z_offset = c_z*scale
    #x_offset = -dx_px/368*c_z
    #y_offset = -dy_px/368*c_z


    x_offset = math.atan(dx_px/alpha_x*c_depth)
    y_offset = math.atan(dy_px/alpha_y*c_depth)
    #x_offset = dx_px/alpha_x*c_depth
    #y_offset = dy_px/alpha_y*c_depth
#     
    scale_x = 1.2
#    scale_y = 1.2
    for point3d in points3d_train:
        x,y,z = point3d
        y+=x_offset #x/y are reversed most of the time !!!!!
        x+=y_offset
        z+=z_offset
        translated_points3d.append((x,y,z))

    alpha = x_offset/c_depth #Small angle approx. sin(alpha)=alpha
    beta = -y_offset/c_depth
    if verbose:
        print(f"alpha: {alpha} beta: {beta} dx_px: {dx_px}, dy_px: {dy_px}")
    #print(alpha,beta)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #rotation = mesh.get_rotation_matrix_from_xyz([beta,alpha,0])
    center_rotation = np.array(mesh.get_rotation_matrix_from_xyz([alpha,beta,0]))
    axis_rotation = np.array(mesh.get_rotation_matrix_from_xyz([0,0,0]))
    
    #center = np.array(translated_points3d[0])


    pcd = o3d.geometry.PointCloud()
    

    #translated_bbox = np.array(translated_points3d)
    #pcd.points = o3d.utility.Vector3dVector(translated_bbox)
    
    offset = -(1-distance_scale) * np.array(points3d_train[0])
    points_3d_offset = np.array(points3d_train) + offset
    old_center = points_3d_offset[0]
    new_center = np.dot(center_rotation, points_3d_offset[0])

    #pcd.points= o3d.utility.Vector3dVector(np.array(points3d_train))
    pcd.points =  o3d.utility.Vector3dVector(points_3d_offset)
    center = np.array([0,0,0])

    pcd.rotate(center_rotation, center)
    #pcd.rotate([beta,alpha,0])
    rotated_points3d = np.asarray(pcd.points)

    rotated_points3d_v2 = points_3d_offset - old_center + new_center
    pcd_v2 = o3d.geometry.PointCloud()
    pcd_v2.points =  o3d.utility.Vector3dVector(rotated_points3d_v2)
    pcd_v2.rotate(axis_rotation, new_center)

    rotated_points3d_v2 = np.asarray(pcd_v2.points)
    #offset =scale * np.array(points3d_train[0])
    #offset =scale * rotated_points3d[0]
    #translated_rotated_points3d=rotated_points3d+offset
    #return translated_points3d
    #return rotated_points3d, points3d_train #translated_points3d
    rotated_points3d = list(map(tuple, rotated_points3d))
    return rotated_points3d_v2, points3d_train
    #return rotated_points3d, points3d_train

def visualize(points3d_query1, points3d_results1, points3d_results2):
    """Small visualisation helper. Interface will change!

    """

    pcd_results1 = o3d.geometry.PointCloud()
    pcd_results2 = o3d.geometry.PointCloud()
    #points3d_train_results1 = points3d_results1
    #points3d_train_rotated, points3d_train_aligned = align_with_bbox_3d(points3d_train, train_bbox, valid_bbox)
    pcd_results1.points = o3d.utility.Vector3dVector(points3d_results1)
    pcd_results2.points = o3d.utility.Vector3dVector(points3d_results2)
    
    #pcd_train_aligned.points = o3d.utility.Vector3dVector(points3d_train_aligned)

    pcd_valid = o3d.geometry.PointCloud()
    pcd_valid.points = o3d.utility.Vector3dVector(points3d_query1)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    bbox_3d_results1 = pcd_results1.get_oriented_bounding_box()
    bbox_3d_results2 = pcd_results2.get_oriented_bounding_box()
    bbox_3d_valid = pcd_valid.get_oriented_bounding_box()
    #bbox_3d.create_from_point_cloud_poisson(pcd_train)
    bbox_3d_results1.color = [0,0,255]
    bbox_3d_results2.color = [255,0,0]
    vis.add_geometry(bbox_3d_results1)
    vis.add_geometry(bbox_3d_results2)
    vis.add_geometry(bbox_3d_valid)
    vis.add_geometry(pcd_results1)
    vis.add_geometry(pcd_results2)
    vis.add_geometry(pcd_valid)

    print(bbox_3d_results1.extent, bbox_3d_results1.center)
    print(bbox_3d_valid.extent, bbox_3d_valid.center)
    

    vis.run()
    ctr = vis.get_view_control()
    pinhole_parameters = ctr.convert_to_pinhole_camera_parameters()
    vis.destroy_window()
    #pinhole_parameters.intrinsic
    #vis.draw_geometries()
    return 

def project_3d_to_2d(points3d, intrinsics):
    """Projects 3d points, (in the Google format) to pixel
    location using the intrinsic matrix. The intrinsic matrix
    must be scaled if the image is scaled.

    Args:
        points3d ([list]): List of 3D points in camera frame.
        intrinsics ([np.array]): Intrinsic parameters of the camera.

    Returns:
        [np.array]: Location of the project points in pixels.
    """
    p3d_fixed = np.array(points3d) #3d points from the dataset. 
    p3d_fixed[:,2]=-p3d_fixed[:,2] #Reverse z axis
    res = np.dot(intrinsics, p3d_fixed.T)
    res=res/res[2]
    x = res[1]
    y = res[0]
    return np.swapaxes(np.vstack([x, y]),0,1)

def get_match_snapped_points(idx, match_idx,  info_df, train_info_df, ground_truth=False, rescale=True, align_axis=True):

    assert type(match_idx)==int
    plane_center, plane_normal = get_plane(info_df, idx)
    points2d, points3d = get_points(info_df,idx)
    points3d_result_rotated, points3d_result = get_match_aligned_points(idx, match_idx, info_df, train_info_df, ground_truth = ground_truth)
    snapped, intersect = snap_box_to_plane(points3d_result_rotated, plane_normal, plane_center, align_axis=align_axis)
    

    result = snapped
    if rescale:
        points2d_result, _ = get_points(train_info_df,match_idx)
        #points3d_result = np.array(points3d_result)
        camera = get_camera(info_df, idx)
        intrinsics = get_intrinsics(camera)
        intrinsics = scale_intrinsics(intrinsics, 0.25, 0.25)
        points2d_px = project_3d_to_2d(snapped, intrinsics)
        dest_bbox = get_bbox(points2d_px, 360, 480)

        points2d_valid, points3d_valid = get_points(info_df, idx)

        valid_image = Image.open(info_df.iloc[idx]["filepath_full"])
        points2d_valid_px = points_2d_to_points2d_px(points2d_valid, valid_image.width, valid_image.height)
        valid_bbox = get_bbox(points2d_valid_px, valid_image.width, valid_image.height)
        scale_x, scale_y = get_scale_factors(dest_bbox, valid_bbox)
        scale_factor = (scale_x + scale_y)/2
        
        #print(scale_factor)
        fixed_point = snapped[0]
        snapped_rotated = snapped.copy()
        #scale_factor = 1.2
        snapped_rotated=(snapped_rotated-fixed_point)*scale_factor+fixed_point
        snapped_rotated, intersect=snap_box_to_plane(snapped_rotated, plane_normal, plane_center)
        result = snapped_rotated

    return result, points3d_result

def get_iou_between_bbox(points_train, points_valid):
    try:
        v_rotated = np.array(points_train)
        v_valid = np.array(points_valid)
        w_rotated = box.Box(vertices=v_rotated)
        w_valid = box.Box(vertices=v_valid)
        loss = iou.IoU(w_rotated, w_valid)
        return float(loss.iou())
    except:
        print("Error computing IoU!")
        return 0

def get_iou(idx:int, embeddings: np.array, info_df: pd.DataFrame, 
            train_embeddings:np.array, train_info_df: pd.DataFrame,
            symmetric=False, rescale=False,
            k=0, show=False, compute_aligned=False, ground_plane=True,
            align_axis=True
            ):
    """Get 3D IoU for a specific query image, using the kth neighbor.

    Args:
        idx (int): Query index in meta dataframe.
        embeddings (np.array): Query embeddings.
        info_df (pd.DataFrame): Query embeddings metadata.
        train_embeddings (np.array): Trainin set embeddings. Will be used for lookup.
        train_info_df (pd.DataFrame): Training set embeddings metadata.
        k (int, optional): kth neighbor to use as result.. Defaults to 0.
        show (bool, optional): Visualize the result in 3D. Defaults to False.

    Returns:
        [type]: [description]
    """
    #@concurrent
    match_idx = find_match_idx(idx, embeddings, train_embeddings,k)
    if ground_plane:
        points3d_processed, points3d_valid = get_match_snapped_points(idx, match_idx, info_df, train_info_df, ground_truth=True, rescale=rescale, align_axis=align_axis)
    else:
        points3d_processed, points3d_valid = get_match_aligned_points(idx, match_idx, info_df, train_info_df, ground_truth=True)
 
    #if show:
    #    visualize(points3d_valid, points3d_train_rotated, points3d_train_aligned)
    iou_value = get_iou_between_bbox(points3d_valid, points3d_processed)
    best_iou = iou_value
    if symmetric:
        for angle in np.arange(5,360,5):
            theta = (angle/180)*np.pi
            pivoted = rotate_bbox_around_its_center(points3d_processed, theta)
            iou_at_theta = get_iou_between_bbox(pivoted, points3d_valid)
            if iou_at_theta > best_iou:
                best_iou = iou_at_theta
        #_get_iou.wait()
    return best_iou , match_idx



def get_iou_rotated(points3d_processed, points3d_valid, initial_iou):
    """Get the IoU metric for a symmetric object by rotating it around its axis.

    Args:
        points3d_processed (np.array): Result Bounding box 
        points3d_valid (np.array): Query Bounding box
        initial_iou (float): Initial IoU when the object is not rotated.

    Returns:
        float: Best IoU obtained while rotating the object around its axis.
    """
    best_iou = initial_iou
    for angle in np.arange(5,360,5):
        theta = (angle/180)*np.pi
        pivoted = rotate_bbox_around_its_center(points3d_processed, theta)
        iou_at_theta = get_iou_between_bbox(pivoted, points3d_valid)
        if iou_at_theta > best_iou:
            best_iou = iou_at_theta
    return best_iou

def find_match_idx(idx, query_embeddings: np.ndarray, train_embeddings:np.ndarray, k=0, score=False):
    """Find the nearest neighbor of a single query embedding in the test set in the training set.

    Args:
        idx (int): Index in the test set.
        query_embeddings (np.ndarray): Test set embeddings.
        train_embeddings (np.ndarray): Train set embeddings.
        k (int, optional): kth neighbor to use. Defaults to 0.

    Returns:
        int: Match index in the train set.
    """
    global use_cupy
    idx = fix_idx(idx)
    lib = cp
    if not use_cupy:
        lib = np
    
    similarity = lib.dot(query_embeddings[idx].T,train_embeddings)
    if k != 0:
        best_matches = lib.argsort(-similarity)
        match_idx = best_matches[k]
    else:
        match_idx = lib.argmin(-similarity)
    if not score:
        return int(match_idx)
    else:
        return int(match_idx), float(similarity[match_idx])

def find_all_match_idx(query_embeddings: np.ndarray, train_embeddings:np.ndarray, k=0):
    """Find all matches for the test set in the training set at the sametime, using cupy.

    This is solely for optimisation purpose in order to get the code to run faster on machines
    with GPU.

    Args:
        query_embeddings (np.ndarray): Test set embeddings.
        train_embeddings (np.ndarray): Train set embeddings.
        k (int, optional): [description]. Defaults to 0.

    Raises:
        ValueError: The case where "k!=0" is not yet implemeted.

    Returns:
        [type]: [description]
    """
    global use_cupy
    print("Using GPU to compute matches!")

    if k != 0:
        #best_matches = cp.argsort(-cp.dot(query_embeddings.T,train_embeddings))
        #match_idx = best_matches[k]
        raise ValueError("The case where k is not 0 must be implemented.")
    else:
        match_idxs = []
        query_chunk_size = 1024
        train_chunk_size = 65536*2
        for i in tqdm(range(0, math.ceil(len(query_embeddings)/query_chunk_size))):
            query_start = i*query_chunk_size
            query_end = query_start + query_chunk_size
            if query_end > len(query_embeddings):
                query_end=len(query_embeddings)
            cuda_query_embeddings = cp.array(query_embeddings[query_start:query_end])

            matches = []
            scores = []
            best_match_idx_chunk_score = np.zeros((query_end-query_start,1))
            best_match_idx_chunk = np.zeros((query_end-query_start,1), dtype=np.uint64)
            for j in range(0,math.ceil(train_embeddings.shape[1]/train_chunk_size)):
                train_start = j*train_chunk_size
                train_end = train_start + train_chunk_size
                if train_end > train_embeddings.shape[1]:
                    train_end=train_embeddings.shape[1]
                cuda_train_embeddings = cp.array(train_embeddings[:,train_start:train_end])
                similarity = cp.dot(cuda_query_embeddings,cuda_train_embeddings)
                match_idx_chunk = cp.argmax(similarity,axis=1)
                match_idx_chunk_score = np.take_along_axis(similarity.get(),np.expand_dims(match_idx_chunk.get(),axis=1),axis=1)
                match_idx_chunk+=train_start
                best_match_idx_chunk = np.where(match_idx_chunk_score>best_match_idx_chunk_score, np.expand_dims(match_idx_chunk.get(), axis=1), best_match_idx_chunk).astype(np.uint64)
                best_match_idx_chunk_score = np.where(match_idx_chunk_score>best_match_idx_chunk_score,match_idx_chunk_score,best_match_idx_chunk_score)
                
                #if use_cupy:
                match_idx_chunk=match_idx_chunk.get()

                matches.append(match_idx_chunk)
            match_idxs+=best_match_idx_chunk.squeeze().tolist()
        return match_idxs

def scale_intrinsics(intrinsics: np.array, scale_x: float, scale_y:float):
    """Scale an intrinsic matrix to simulate a lower resolution sensor. This is 
    useful for scene reconstruction when the original image was taken at a resolution
    and the analysed image is in another resolution.

    Args:
        intrinsics (np.array): Camera intrinsic matrix.
        scale_x (float): Scaling factor for x axis.
        scale_y (float): Scaling factor for y axis
    """
    scaled_intrinsics = intrinsics.copy()
    scaled_intrinsics[0,:]*=scale_x
    scaled_intrinsics[1,:]*=scale_y
    return scaled_intrinsics

def get_intrinsics(camera):
    return np.array(camera.intrinsics).reshape(3,3)

def describe_intrinsics(intrinsics: np.array):
    """Describe the intrinsics matrix in their common names.

    Args:
        intrinsics (np.array): Intrinsics
    """
    alpha_x = intrinsics[0,0]
    alpha_y = intrinsics[1,1]
    center_x = intrinsics[0,2]
    center_y = intrinsics[1,2]
    return float(alpha_x), float(alpha_y), float(center_x), float(center_y)  

embeddings = None
info_df = None
train_embeddings = None
train_info_df = None
#ground_plane = True
#symmetric = False
#rescale = False
args = None
all_match_idxs = None
import argparse
import time

def get_subset(info_df, embeddings, ratio):
    if ratio==1:
        return info_df, embeddings
    info_df["video_uid"]=info_df["category"]+"_"+info_df["video_id"]
    video_uids = list(info_df["video_uid"].unique())
    len(video_uids)
    #ratio = 0.1
    videos_uids_subset = random.sample(video_uids, int(len(video_uids)*ratio))
    print(f"Using a subset of {len(videos_uids_subset)} out of {len(video_uids)} total sequences for evaluation.")

    info_df_subset =info_df[info_df["video_uid"].isin(videos_uids_subset)]
    #info_df_subset
    embeddings_subset=embeddings[:,list(info_df_subset.index)]
    #embeddings_subset.shape
    info_df_subset = info_df_subset.reset_index()
    return info_df_subset, embeddings_subset

def get_iou_mp(idx:int #, symmetric=False, rescale=False,
            #k=0, show=False, compute_aligned=False, ground_plane=True,
            ):
    """Get 3D IoU for a specific query image, using the kth neighbor.

    Args:
        idx (int): Query index in meta dataframe.
        embeddings (np.array): Query embeddings.
        info_df (pd.DataFrame): Query embeddings metadata.
        train_embeddings (np.array): Trainin set embeddings. Will be used for lookup.
        train_info_df (pd.DataFrame): Training set embeddings metadata.
        k (int, optional): kth neighbor to use as result.. Defaults to 0.
        show (bool, optional): Visualize the result in 3D. Defaults to False.

    Returns:
        [type]: [description]
    """
    
    global args, info_df, embeddings, train_info_df, train_embeddings, all_match_idxs
    category = info_df.iloc[idx]["category"]

    #all_match_idxs = find_match_idx(idx, embeddings, train_embeddings,0)
    match_idx=all_match_idxs[idx]
    if args.random_bbox:
        match_idx = random.randint(0,len(train_info_df))
    if args.random_bbox_same:
        match_idx = random.sample(list(train_info_df.query(f"category=='{category}'").index),1)
        match_idx = match_idx[0]

    if args.legacy:
        if args.ground_plane:
            points3d_processed, points3d_valid = get_match_snapped_points(idx, match_idx, info_df, train_info_df, ground_truth=True, rescale=args.rescale, align_axis=not args.no_align_axis)
        else:
            points3d_processed, points3d_valid = get_match_aligned_points(idx, match_idx, info_df, train_info_df, ground_truth=True)
    else:
        _, points3d_valid = get_points(info_df, idx)
        points3d_processed = get_bounding_box(idx, match_idx, info_df, train_info_df, adjust_scale=True)
    #if show:
    #    visualize(points3d_valid, points3d_train_rotated, points3d_train_aligned)
    iou_value = get_iou_between_bbox(points3d_valid, points3d_processed)
    best_iou = iou_value
    if args.symmetric:        
        if category=="cup" or category=="bottle":
            best_iou = get_iou_rotated(points3d_processed, points3d_valid, best_iou)
        #_get_iou.wait()
    return best_iou , idx, match_idx
#@synchronized
def main():
    global args, info_df, embeddings, train_info_df, train_embeddings, ground_plane, symmetric, rescale, all_match_idxs, use_cupy
    """Command line tool for evaluating zero shot pose estimation
    on objectron."""
    parser = argparse.ArgumentParser(description='Command line tool for evaluating zero shot pose estimation on objectron.')
    parser.add_argument("experiment", type=str, help="Experiment folder location. i.e. ../SimSiam/outputs/objectron_96x96_experiment_synchflip_next")
    parser.add_argument("--subset_size", type=int, default=1000, help="Number of samples for 3D IoU evaluation")
    parser.add_argument("--ground_plane", default=True, help="If enabled, snap to ground plane")
    parser.add_argument("--iou_t", default=0.5, help="IoU threshold required to consider a positive match")
    parser.add_argument("--symmetric", action="store_true",help="Rotate symetric objects (cups, bottles) and keep maximum IoU.")
    parser.add_argument("--rescale", action="store_true",help="Rescale 3d bounding box")
    parser.add_argument("--random_bbox", action="store_true", help="Fit a randomly selected bbox instead of the nearest neighbor")
    parser.add_argument("--random_bbox_same", action="store_true", help="Fit a randomly selected bbox from same category instead of the nearest neighbor")
    parser.add_argument("--trainset_ratio", type=float, default=1, help="Ratio of the training set sequences used for inference")
    parser.add_argument("--single_thread", action="store_true", help="Disable multithreading.")
    parser.add_argument("--cpu", action="store_true", help="Disable cuda accelaration.")
    parser.add_argument("--no_align_axis", action="store_true", help="Don't to to align axis with ground plane.")
    parser.add_argument("--legacy", action="store_true", help="Deprecated legacy evalution mode")
    args = parser.parse_args()
    symmetric = args.symmetric
    rescale = args.rescale
    if args.cpu:
        use_cupy=False
    embeddings, info_df, train_embeddings, train_info_df = read_experiment(args.experiment)
    all_match_idxs = find_all_match_idx(embeddings, train_embeddings, 0)
    get_iou_mp(1)
    if args.subset_size < len(info_df):
        subset = random.sample(range(0,len(info_df)),args.subset_size)
    else:
        print(f"Evaluating on all samples size subset size ({args.subset_size}) is larger that test set size ({len(info_df)}).")
        subset = list(range(0,len(info_df)))
        random.shuffle(subset)
    if args.trainset_ratio > 0 and args.trainset_ratio <= 1:
        train_info_df, train_embeddings = get_subset(train_info_df, train_embeddings, args.trainset_ratio)
    else:
        raise ValueError("Training set ratio must be between 0 and 1!")
    ious = {}
    results = []
    ious_aligned = []
    threshold = args.iou_t
    valid_count = 0
    get_iou_params = []

    for idx in subset:
        #params = (idx, args.symmetric, args.rescale)
        get_iou_params.append(idx)
    if not args.single_thread:
        with Pool(6) as p:
            results = list(tqdm(p.imap(get_iou_mp, get_iou_params), total=len(subset)))
    else:
        for idx in tqdm(get_iou_params):
            results.append(get_iou_mp(idx))

    for iou, idx, match_idx in results:
        meta = info_df.iloc[idx]
        category = meta["category"]
        if not category in ious:
            ious[category]=[]
        #symmetric = False
        #if category=="cup" or category=="bottle":
        #    symmetric=True
        #results[idx] = get_iou(idx, embeddings, info_df, train_embeddings, train_info_df, symmetric=args.symmetric, rescale=args.rescale)
        #iou, match_idx= get_iou(idx, embeddings, info_df, train_embeddings, train_info_df, symmetric=args.symmetric, rescale=args.rescale)
        if iou > threshold:
            valid_count+=1
        ious[category].append(float(iou))
    #get_iou.wait()
    ious = pd.DataFrame.from_dict(ious, orient='index').T
    ious.to_csv("raw_results.txt")
    print(f"{'category' : <10}\tmean iou\tmedian iou\tAP at iou_t")
    for category in sorted(ious.columns):
        column = ious[category]
        ap_at_iout=(column > args.iou_t).sum()/column.notnull().sum()
        print(f"{category: <10}\t{column.mean():0.2f}\t\t{column.median():0.2f}\t\t{ap_at_iout:0.2f}")


    #ious = np.array(ious)
    #median = np.median(ious)
    #mean = ious.mean()
    #valid_percentage = valid_count/len(subset) * 100
    #print(f"Median IoU: {median}, Mean IoU: {mean}, Valid at {args.iou_t:0.2f} IoU: {valid_percentage:0.2f}%")

if __name__ == "__main__":
    main()
