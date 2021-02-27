import typing

import cv2 as cv
import numpy as np

import selfsupmotion.data.objectron.dataset.graphics
import selfsupmotion.data.utils


def display_debug_frames(
        sample_frame: typing.Dict,
):
    cv.imshow("raw", sample_frame["IMAGE"])
    assert (sample_frame["POINT_NUM"] == 9).all() and sample_frame["INSTANCE_NUM"] > 0
    points_2d = sample_frame["POINT_2D"].reshape((sample_frame["INSTANCE_NUM"], 9, 3))
    image = np.copy(sample_frame["IMAGE"])
    for instance_idx in range(sample_frame["INSTANCE_NUM"]):
        image = selfsupmotion.data.objectron.dataset.graphics.draw_annotation_on_image(
            image, points_2d[instance_idx, :, :], [9])
        cv.circle(image, tuple(sample_frame["CENTROID_2D_IM"][instance_idx]), 3, (255, 255, 255), -1)
    cv.imshow("2d", image)
    points_3d = sample_frame["POINT_3D"].reshape((sample_frame["INSTANCE_NUM"], 9, 3))
    view_matrix = sample_frame["VIEW_MATRIX"].reshape((4, 4))
    proj_matrix = sample_frame["PROJECTION_MATRIX"].reshape((4, 4))
    im_width, im_height = image.shape[1], image.shape[0]
    image = np.copy(sample_frame["IMAGE"])
    for instance_idx in range(sample_frame["INSTANCE_NUM"]):
        curr_points_3d = points_3d[instance_idx]
        points_2d_new = selfsupmotion.data.utils.project_points(
            curr_points_3d, proj_matrix, view_matrix, im_width, im_height)
        for point_id in range(points_2d_new.shape[0]):
            cv.circle(image, (points_2d_new[point_id, 0], points_2d_new[point_id, 1]), 10,
                      (0, 255, 0), -1)
    cv.imshow("3d", image)
    cv.waitKey(0)
