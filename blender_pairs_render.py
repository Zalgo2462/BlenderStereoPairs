"""
Generates a stereo pair dataset using Blender with images, calibration details, and depth maps.
The generated camera positions are evenly sampled across a spherical segment, with each camera pointing
towards the center of the sphere. Each pair of cameras are considered a stereo pair if the central angle
between the two cameras as measured from the center of the sphere is within a given threshold.

Run with `blender <blend file> -P blender_pairs_render.py -- [args]`.
Unless `--output_dir <directory>` is passed as part of `args`, the results will be written to `./blender_pairs`.

The list of stereo pairs can be found in `blender_pairs/pairs.csv`. Each row represents a stereo pair
with the format `<image 1 camera name>, <image 1 frame number>, <image 2 camera name>, <image 2 frame number>`.
Each stereo pair addresses two images, depth maps, intrinsic calibration matrices, and extrinsic calibration matrices.
The image data and depth maps for each (camera, frame) tuple can be found in EXR format at
 `blender_pairs/EXR/<camera name>/<frame number>.exr`. The RGB-D data is stored as 32 bit floats in the channels
 `Image.R`, `Image.G`, `Image.B`, and `Depth.V`.
The 3x3 intrinsic calibration matrix for each camera can be found in numpy txt format at
 `blender_pairs/K/<camera name>.txt`.
The 3x4 extrrinsic calibration matrix for each (camera, frame) tuple can be found in numpy txt format at
 `blender_pairs/RT/<camera name>/<frame number>.txt`.
The arguments passed to the script are stored in `blender_paris/params.json`.

"""
import os
import shutil
import sys
import json
from argparse import ArgumentParser

import numpy as np
import scipy.spatial

import bpy
import mathutils as mutils
import tempfile


def define_args(parent_parser: ArgumentParser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # Camera objects to render from
    # These cameras should have the "Copy Location" (with Offset) and "Track To" Blender modifiers set up
    # to track a single Blender object. This ensures each image will be focused on the same position in space.
    # Additionally, this allows the code to use relative coordinates for the cameras and skip handling
    # intrinsic camera rotations.
    parser.add_argument("--camera", type=str, default="Camera.001")

    # The number of camera positions on the truncated sphere to sample
    parser.add_argument("--num_images", type=int, default=300)  # 1000

    # How far away the cameras are from the focus object
    parser.add_argument("--radius", type=float, default=2)  # 2-2.39

    # How far up and down the cameras should be placed along the phi angle
    # in degrees as measured from the top of the sphere down.
    # Controls the truncation of the sphere.
    parser.add_argument("--min_phi", type=float, default=0)
    parser.add_argument("--max_phi", type=float, default=90)

    # Maximum central angle between two cameras to be considered a pair (in degrees)
    parser.add_argument("--max_central_angle", type=float, default=45)  # 45

    parser.add_argument("--output_dir", type=str, default="blender_pairs")
    return parser


def validate_args(args):
    if args.num_images < 1:
        return False
    if args.min_phi > args.max_phi:
        return False
    if args.radius <= 0:
        return False
    try:
        _ = bpy.data.objects[args.camera]
    except KeyError:
        return False
    return True


def sample_fibonacci_sphere(num_samples, min_phi, max_phi):
    # find linear bounds in the unit interval for the phi bounds
    min_linear_intvl = (1 - np.cos(min_phi)) / 2
    max_linear_intvl = (1 - np.cos(max_phi)) / 2

    # divide the interval up into num_samples + 1 even sections
    unit_intvl = np.linspace(min_linear_intvl, max_linear_intvl, num=num_samples + 1)
    # find the midpoints of each of the even sections and throw away the last point to get num_samples
    unit_intvl += (unit_intvl[1] - unit_intvl[0])
    unit_intvl = unit_intvl[:-1]
    phi = np.arccos(1 - 2 * unit_intvl)

    theta = np.pi * (1 + np.sqrt(5)) * (np.arange(0, num_samples, dtype=float) + 0.5)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))


def find_camera_pairs(camera_name, camera_locs, angle_threshold=(np.pi / 4)):
    dists = np.arccos(-1 * scipy.spatial.distance.pdist(camera_locs, metric="cosine") + 1)
    dists = scipy.spatial.distance.squareform(dists)
    pairs = []
    for i in range(camera_locs.shape[0] - 1):
        for j in range(i + 1, camera_locs.shape[0]):
            if dists[i, j] <= angle_threshold:
                pairs.append((camera_name, i, camera_name, j))
    return pairs


def render_depth_rgb(view_name):
    render_layer = bpy.context.window.view_layer

    bpy.context.scene.render.engine = "CYCLES"
    render_layer.use_ao = True
    render_layer.use_solid = True
    render_layer.use_strand = False
    render_layer.use_sky = False
    render_layer.use_pass_z = True
    render_layer.use_pass_combined = True

    # switch on nodes
    bpy.context.scene.use_nodes = True

    # clear other nodes
    bpy.context.scene.node_tree.nodes.clear()

    # create render layer node to get the resulting render layers
    render_layers_node = bpy.context.scene.node_tree.nodes.new("CompositorNodeRLayers")

    img_saver_node = bpy.context.scene.node_tree.nodes.new("CompositorNodeOutputFile")
    img_saver_node.base_path = view_name
    img_saver_node.format.file_format = "OPEN_EXR_MULTILAYER"
    img_saver_node.format.color_depth = '32'
    img_saver_node.file_slots.new("Depth")
    img_saver_node.format.color_mode = "RGB"

    bpy.context.scene.node_tree.links.new(render_layers_node.outputs.get("Image"), img_saver_node.inputs.get("Image"))
    bpy.context.scene.node_tree.links.new(render_layers_node.outputs.get("Depth"), img_saver_node.inputs.get("Depth"))

    bpy.ops.render.render()
    return


def camera_intrinsics_matrix(camera_data):
    # https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    f_mm = camera_data.lens
    x_res_px = bpy.context.scene.render.resolution_x
    y_res_px = bpy.context.scene.render.resolution_y
    render_scale = bpy.context.scene.render.resolution_percentage / 100.0
    aspect_ratio = bpy.context.scene.render.pixel_aspect_x / bpy.context.scene.render.pixel_aspect_y

    if camera_data.sensor_fit == 'VERTICAL':
        s_u = x_res_px / camera_data.sensor_width * (1.0 / aspect_ratio) * render_scale
        s_v = y_res_px / camera_data.sensor_height * render_scale
    else:  # 'HORIZONTAL', 'AUTO;
        s_u = x_res_px / camera_data.sensor_width * render_scale
        s_v = y_res_px / camera_data.sensor_height * aspect_ratio * render_scale

    f_u = f_mm * s_u
    f_v = f_mm * s_v
    u_0 = x_res_px / 2.0 * render_scale
    v_0 = y_res_px / 2.0 * render_scale
    skew = 0

    return np.array([
        [f_u, skew, u_0],
        [0.0, f_v, v_0],
        [0.0, 0.0, 1.0]
    ])


def camera_extrinsics_matrix(camera):
    pose_location, pose_rotation_q = camera.matrix_world.decompose()[0:2]

    inv_rotation = pose_rotation_q.to_matrix().transposed()
    inv_translation = -1 * inv_rotation @ pose_location

    blender_2_cv_rotation = mutils.Matrix((
        (1, 0, 0),
        (0, -1, 0),
        (0, 0, -1)
    ))

    extrinsic_rotation = blender_2_cv_rotation @ inv_rotation
    extrinsic_translation = blender_2_cv_rotation @ inv_translation

    extrinsic_matrix = np.array([
        [extrinsic_rotation[i][j] for j in range(3)] + [extrinsic_translation[i]] for i in range(3)
    ])

    return extrinsic_matrix


def make_output_directories(output_dir, camera_names):
    os.makedirs(output_dir, exist_ok=True)
    exr_dir = os.path.join(output_dir, "EXR")
    os.makedirs(exr_dir, exist_ok=True)
    intrinsics_dir = os.path.join(output_dir, "K")
    os.makedirs(intrinsics_dir, exist_ok=True)
    extrinsics_dir = os.path.join(output_dir, "RT")
    os.makedirs(extrinsics_dir, exist_ok=True)

    exr_dirs = {}
    extrinsics_dirs = {}
    for camera_name in camera_names:
        exr_dirs[camera_name] = os.path.join(exr_dir, camera_name)
        os.makedirs(exr_dirs[camera_name], exist_ok=True)

        extrinsics_dirs[camera_name] = os.path.join(extrinsics_dir, camera_name)
        os.makedirs(extrinsics_dirs[camera_name], exist_ok=True)

    return exr_dirs, intrinsics_dir, extrinsics_dirs


def write_camera_pairs(filename, pairs):
    with open(filename, 'w') as f:
        f.writelines((",".join(str(y) for y in x) + "\n" for x in pairs))
    return


def main():
    parser = ArgumentParser(add_help=True)
    parser = define_args(parser)
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    if not validate_args(args):
        print("Invalid arguments")
        sys.exit(1)

    # We only use one camera in this dataset
    camera_name = args.camera

    print("Writing results to %s" % args.output_dir)

    # Dir structure:
    # Top level: EXR, K, RT
    # EXR and RT each contain directories named for each camera
    # K contains a numpy txt file named for each camera
    # exr_dirs is a dict from camera name to output/EXR/camera_name/ directories
    # intrinsics_dir is the output/K/ directory
    # extrinsincs_dirs is a dict from camera name to output/RT/camera_name/ directories
    exr_dirs, intrinsics_dir, extrinsics_dirs = make_output_directories(
        args.output_dir, [camera_name]
    )

    with open(os.path.join(args.output_dir, "params.json"), 'w') as fd:
        json.dump(args.__dict__, fd, indent=2)

    print(
        "Generating %d evenly spaced camera locations on the unit sphere such that phi is within (%0.2f, %0.2f)" % (
            args.num_images, args.min_phi, args.max_phi
        )
    )

    # sample sphere within phi range
    min_phi = args.min_phi / 180.0 * np.pi
    max_phi = args.max_phi / 180.0 * np.pi
    camera_locs = sample_fibonacci_sphere(args.num_images, min_phi, max_phi)

    print("Determining camera locations within %0.2f degrees of each other" % args.max_central_angle)
    # save pairs of camera locations which are close by great circle distance
    max_central_angle = args.max_central_angle / 180.0 * np.pi
    camera_pairs = find_camera_pairs(camera_name, camera_locs, max_central_angle)
    write_camera_pairs(os.path.join(args.output_dir, "pairs.csv"), camera_pairs)
    print("Found %d pairs" % len(camera_pairs))

    # set the camera as active
    camera = bpy.data.objects.get(camera_name)
    bpy.context.scene.camera = camera

    print("Saving camera intrinsics")
    # save the camera intrinsic parameters
    camera_k = camera_intrinsics_matrix(camera.data)
    np.savetxt(os.path.join(intrinsics_dir, camera_name + ".txt"), camera_k)

    # for rendering. Rendering directly to the output directory will lead
    # to name collisions due to Blender being ... difficult
    temp_path = tempfile.mkdtemp()

    print("Rendering images and saving extrinsics")
    for image_idx in range(camera_locs.shape[0]):
        # get the next camera location and assign it to the camera
        location = camera_locs[image_idx]
        location *= args.radius
        camera.location = mutils.Vector(tuple(location))
        bpy.context.window.view_layer.update()  # force location change to take.

        # given the camera constraints in Blender, the camera should already
        # be oriented correctly

        # save the extrinsics for this location
        camera_rt = camera_extrinsics_matrix(camera)
        np.savetxt(os.path.join(extrinsics_dirs[camera_name], "%04d.txt" % image_idx), camera_rt)

        # render the image
        render_depth_rgb(temp_path + os.path.sep)
        shutil.move(os.path.join(temp_path, "0001.exr"), os.path.join(exr_dirs[camera_name], "%04d.exr" % image_idx))
        print(".", end="", flush=True)

    os.rmdir(temp_path)


main()
