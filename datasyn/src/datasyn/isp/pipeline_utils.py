"""
Camera pipeline utilities.
"""

from typing import Any

import cv2
import exifread
import jax.numpy as jnp
import numpy as np
import rawpy
from exifread.utils import Ratio
from numpy.typing import NDArray

from datasyn.isp.demosaic import demosaicing_CFA_Bayer_Menon2007
from datasyn.jaxutils import jx
from datasyn.jaxutils.typing import JArray


def get_visible_raw_image(image_path) -> tuple[NDArray, Any]:
    raw = rawpy.imread(str(image_path))
    raw_image = raw.raw_image_visible.copy()
    return raw_image, raw


def get_image_tags(image_path):
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f)
    return tags


def get_metadata(image_path, raw):
    metadata = _get_metadata_rawpy(image_path, raw)

    if metadata["black_level"] is None:
        metadata["black_level"] = 0
        print("Black level is None; using 0.")
    if metadata["white_level"] is None:
        metadata["white_level"] = 2**16
        print("White level is None; using 2 ** 16.")
    if metadata["cfa_pattern"] is None:
        metadata["cfa_pattern"] = [0, 1, 1, 2]
        print("CFAPattern is None; using [0, 1, 1, 2] (RGGB)")
    if metadata["as_shot_neutral"] is None:
        metadata["as_shot_neutral"] = [1, 1, 1]
        print("AsShotNeutral is None; using [1, 1, 1]")
    if metadata["color_matrix_1"] is None:
        metadata["color_matrix_1"] = [1] * 9
        print("ColorMatrix1 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata["color_matrix_2"] is None:
        metadata["color_matrix_2"] = [1] * 9
        print("ColorMatrix2 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata["orientation"] is None:
        metadata["orientation"] = 0
        print("Orientation is None; using 0.")
    return metadata


def _get_metadata_rawpy(image_path, raw):
    metadata = {}

    metadata["linearization_table"] = None

    bl = raw.black_level_per_channel
    metadata["black_level"] = [float(x) for x in bl]
    metadata["white_level"] = float(raw.white_level)

    raw_pattern = raw.raw_pattern  # shape (2,2)
    color_desc = raw.color_desc
    if isinstance(color_desc, bytes):
        color_desc = color_desc.decode("ascii")

    letter_to_dng = {"R": 0, "G": 1, "B": 2}
    raw_to_dng = {}
    for i, ch in enumerate(color_desc):
        raw_to_dng[i] = letter_to_dng.get(ch, 1)

    pattern_dng = np.zeros_like(raw_pattern)
    for y in range(raw_pattern.shape[0]):
        for x in range(raw_pattern.shape[1]):
            pattern_dng[y, x] = raw_to_dng[int(raw_pattern[y, x])]

    metadata["cfa_pattern"] = pattern_dng.flatten().tolist()

    gains = np.array(raw.camera_whitebalance[:3], dtype=float)
    gains /= gains[1]
    asn = 1.0 / gains
    metadata["as_shot_neutral"] = asn.tolist()

    try:
        rgb_xyz = np.asarray(raw.rgb_xyz_matrix, dtype=float)
        cam2xyz = rgb_xyz[:3, :3]
        cm = cam2xyz.flatten().tolist()
        metadata["color_matrix_1"] = cm
        metadata["color_matrix_2"] = cm
    except Exception:
        metadata["color_matrix_1"] = [1.0] * 9
        metadata["color_matrix_2"] = [1.0] * 9
        print("rgb_xyz_matrix not available; using identity color matrices.")

    tags = get_image_tags(image_path)
    orientation = get_orientation(tags, None)
    if orientation is None:
        orientation = 1
        print("Orientation not found; using 1 (no rotation).")
    metadata["orientation"] = orientation

    metadata["noise_profile"] = None
    metadata["opcode_lists"] = []

    tc = raw.tone_curve
    metadata["tone_curve"] = tc / (len(tc) - 1)

    return metadata


def get_orientation(tags, ifds):
    possible_tags = ["Orientation", "Image Orientation"]
    return get_values(tags, possible_tags)


def get_values(tags, possible_keys):
    values = None
    for key in possible_keys:
        if key in tags.keys():
            values = tags[key].values
    return values


def normalize(raw_image, black_level, white_level):
    if type(black_level) is list and len(black_level) == 1:
        black_level = float(black_level[0])
    if type(white_level) is list and len(white_level) == 1:
        white_level = float(white_level[0])
    black_level_mask = black_level
    if type(black_level) is list and len(black_level) == 4:
        if type(black_level[0]) is Ratio:
            black_level = ratios2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0] :: step2, idx[1] :: step2] = black_level[i]
    normalized_image = raw_image.astype(np.float32) - black_level_mask
    normalized_image[normalized_image < 0] = 0
    normalized_image = normalized_image / (white_level - black_level_mask)
    return normalized_image


def ratios2floats(ratios):
    floats = []
    for ratio in ratios:
        floats.append(float(ratio.num) / ratio.den)
    return floats


def lens_shading_correction(
    raw_image, gain_map_opcode, bayer_pattern, gain_map=None, clip=True
):
    if gain_map is None and gain_map_opcode:
        gain_map = gain_map_opcode.data["map_gain_2d"]

    gain_map = cv2.resize(
        gain_map,
        dsize=(raw_image.shape[1] // 2, raw_image.shape[0] // 2),
        interpolation=cv2.INTER_LINEAR,
    )
    if len(gain_map.shape) == 2:
        gain_map = np.tile(gain_map[..., np.newaxis], [1, 1, 4])

    if gain_map_opcode:
        top = gain_map_opcode.data["top"]
        left = gain_map_opcode.data["left"]
        bottom = gain_map_opcode.data["bottom"]
        right = gain_map_opcode.data["right"]
        rp = gain_map_opcode.data["row_pitch"]
        cp = gain_map_opcode.data["col_pitch"]

    result_image = raw_image.copy()

    upper_left_idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    bayer_pattern_idx = np.array(bayer_pattern)
    bayer_pattern_idx[bayer_pattern_idx == 2] = 3
    if bayer_pattern_idx[3] == 1:
        bayer_pattern_idx[3] = 2
    else:
        bayer_pattern_idx[2] = 2
    for c in range(4):
        i0 = upper_left_idx[c][0]
        j0 = upper_left_idx[c][1]
        result_image[i0::2, j0::2] *= gain_map[:, :, bayer_pattern_idx[c]]

    if clip:
        result_image = np.clip(result_image, 0.0, 1.0)

    return result_image


def white_balance(normalized_image, as_shot_neutral, cfa_pattern):
    as_shot_neutral = np.asarray(as_shot_neutral, dtype=np.float32)

    idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    step2 = 2
    white_balanced_image = np.zeros(normalized_image.shape)
    for i, idx in enumerate(idx2by2):
        idx_y = idx[0]
        idx_x = idx[1]
        white_balanced_image[idx_y::step2, idx_x::step2] = (
            normalized_image[idx_y::step2, idx_x::step2]
            / as_shot_neutral[cfa_pattern[i]]
        )
    white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
    return white_balanced_image


def get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type="VNG"):
    if alg_type != "":
        alg_type = "_" + alg_type
    if output_channel_order == "BGR":
        if cfa_pattern == [0, 1, 1, 2]:
            opencv_demosaic_flag = eval("cv2.COLOR_BAYER_BG2BGR" + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:
            opencv_demosaic_flag = eval("cv2.COLOR_BAYER_RG2BGR" + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:
            opencv_demosaic_flag = eval("cv2.COLOR_BAYER_GB2BGR" + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:
            opencv_demosaic_flag = eval("cv2.COLOR_BAYER_GR2BGR" + alg_type)
        else:
            opencv_demosaic_flag = eval("cv2.COLOR_BAYER_BG2BGR" + alg_type)
            print("CFA pattern not identified.")
    else:  # RGB
        if cfa_pattern == [0, 1, 1, 2]:
            opencv_demosaic_flag = eval("cv2.COLOR_BAYER_BG2RGB" + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:
            opencv_demosaic_flag = eval("cv2.COLOR_BAYER_RG2RGB" + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:
            opencv_demosaic_flag = eval("cv2.COLOR_BAYER_GB2RGB" + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:
            opencv_demosaic_flag = eval("cv2.COLOR_BAYER_GR2RGB" + alg_type)
        else:
            opencv_demosaic_flag = eval("cv2.COLOR_BAYER_BG2RGB" + alg_type)
            print("CFA pattern not identified.")
    return opencv_demosaic_flag


def demosaic(
    white_balanced_image, cfa_pattern, output_channel_order="BGR", alg_type="menon2007"
):
    if alg_type == "VNG":
        max_val = 255
        wb_image = (white_balanced_image * max_val).astype(np.uint8)
    else:
        max_val = 1.0
        wb_image = white_balanced_image.astype(np.float32)

    if alg_type in ["", "EA", "VNG"]:
        opencv_demosaic_flag = get_opencv_demsaic_flag(
            cfa_pattern, output_channel_order, alg_type=alg_type
        )
        demosaiced_image = (
            cv2.cvtColor(
                np.clip(
                    65535 * np.asarray(white_balanced_image, dtype=np.float32), 0, 65535
                ).astype(jnp.uint16),
                opencv_demosaic_flag,
            )
            / 65535
        )

    elif alg_type == "menon2007":
        cfa_pattern_str = "".join(["RGB"[i] for i in cfa_pattern])
        demosaiced_image = demosaicing_CFA_Bayer_Menon2007(
            wb_image, pattern=cfa_pattern_str
        )
        demosaiced_image = np.asarray(demosaiced_image)

    demosaiced_image = demosaiced_image.astype(dtype=np.float32) / max_val

    return demosaiced_image


@jx.jit
def _temp_jax_cam_to_xyz(demosaiced: JArray, xyz2cam1: JArray):
    xyz2cam1 = jnp.asarray(xyz2cam1)

    row_sum = jnp.sum(xyz2cam1, axis=1, keepdims=True)
    xyz2cam1 = xyz2cam1 / row_sum

    cam2xyz1 = jnp.linalg.inv(xyz2cam1)

    xyz_image = demosaiced @ cam2xyz1.T
    return xyz_image


def apply_color_space_transform(demosaiced_image, color_matrix_1, color_matrix_2):
    xyz2cam1 = np.reshape(np.asarray(color_matrix_1, dtype=np.float32), (3, 3))
    return _temp_jax_cam_to_xyz(demosaiced_image, xyz2cam1)


@jx.jit
def _temp_jax_xyz_to_srgb(xyz_image: JArray):
    xyz2srgb = jnp.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=xyz_image.dtype,
    )

    row_sum = jnp.sum(xyz2srgb, axis=-1, keepdims=True)
    xyz2srgb = xyz2srgb / row_sum

    srgb_image = xyz_image @ xyz2srgb.T
    srgb_image = jnp.clip(srgb_image, 0.0, 1.0)
    return srgb_image


def transform_xyz_to_srgb(xyz_image):
    return _temp_jax_xyz_to_srgb(xyz_image)


def fix_orientation(image, orientation):
    if type(orientation) is list:
        orientation = orientation[0]

    image = np.asarray(image)

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def reverse_orientation(image, orientation):
    rev_orientations = np.array([1, 2, 3, 4, 5, 8, 7, 6])
    return fix_orientation(image, rev_orientations[orientation - 1])
