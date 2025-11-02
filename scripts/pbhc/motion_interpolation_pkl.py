"""
Interpolate motion data with specified start and end frames.

Weiji Xie: @2025.11.02
- This is a 29 dof version, not the 23 dof version in PBHC repo.

"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def convert_pkl(data, output_filename, contact_mask):
    """Convert motion data to pkl format."""
    data = data.astype(np.float32)
    root_trans = data[:, :3]
    root_qua = data[:, 3:7]
    dof_new = data[:, 7:]

    data_dump = {
        "root_trans_offset": root_trans,
        "dof": dof_new,
        "root_rot": root_qua,
        "fps": 30,
        "contact_mask": contact_mask,
    }

    print("Output DOF Shape", dof_new.shape)
    print("Output Filename: ", output_filename + ".pkl")
    joblib.dump({output_filename: data_dump}, f"{output_filename}.pkl")


def _interpolate_root_rot(rot_aa, default_rr_aa, num_frames, reverse=False):
    """Interpolate root rotation using SLERP."""
    rotations = R.from_euler(
        "ZYX",
        [
            np.concatenate((rot_aa[0:1], default_rr_aa[1:])),
            np.concatenate((rot_aa[0:1], rot_aa[1:])),
        ],
    )
    times = np.linspace(1, 0, num_frames) if reverse else np.linspace(
        0, 1, num_frames)
    slerp = Slerp([0, 1], rotations)
    interp_rots = slerp(times).as_euler("ZYX")
    return R.from_euler("ZYX", interp_rots).as_quat()


def _create_extended_trans(root_trans_frame,
                           default_z,
                           num_frames,
                           is_start=True):
    """Create extended root translation (only Z-axis interpolated)."""
    frame_z = root_trans_frame[2]
    z_values = np.linspace(default_z, frame_z,
                           num_frames) if is_start else np.linspace(
                               frame_z, default_z, num_frames)
    extended_trans = np.zeros((num_frames, 3))
    extended_trans[:, 0] = root_trans_frame[0]
    extended_trans[:, 1] = root_trans_frame[1]
    extended_trans[:, 2] = z_values
    return extended_trans


def interpolate_motion(
    input_data,
    start_ext_frames,
    end_ext_frames,
    default_pose,
    output_pkl,
    contact_mask,
    fix_root_rot,
):
    """Interpolate motion data at start and end with default pose."""
    root_trans = input_data[:, :3]
    root_rot = input_data[:, 3:7]
    dof_pos = input_data[:, 7:]

    default_rt = default_pose[0:3]
    default_rr = default_pose[3:7]
    default_dof = default_pose[7:]
    default_rr_aa = R.from_quat(default_rr).as_euler("ZYX")

    # 起始处插值
    start_root_trans = np.zeros((0, 3))
    start_rr = np.zeros((0, 4))
    start_dof = np.zeros((0, 29))

    if start_ext_frames > 0:
        start_rot_aa = R.from_quat(root_rot[0]).as_euler("ZYX")
        start_root_trans = _create_extended_trans(root_trans[0],
                                                  default_rt[2],
                                                  start_ext_frames,
                                                  is_start=True)
        start_dof = np.linspace(default_dof,
                                dof_pos[0],
                                num=start_ext_frames + 1,
                                endpoint=False)[1:].reshape(-1, 29)
        if not fix_root_rot:
            start_rr = _interpolate_root_rot(start_rot_aa,
                                             default_rr_aa,
                                             start_ext_frames,
                                             reverse=False)

    # 结束处插值
    end_root_trans = np.zeros((0, 3))
    end_rr = np.zeros((0, 4))
    end_dof = np.zeros((0, 29))

    if end_ext_frames > 0:
        end_rot_aa = R.from_quat(root_rot[-1]).as_euler("ZYX")
        end_root_trans = _create_extended_trans(root_trans[-1],
                                                default_rt[2],
                                                end_ext_frames,
                                                is_start=False)
        end_dof = np.linspace(dof_pos[-1], default_dof,
                              num=end_ext_frames + 1)[1:].reshape(-1, 29)
        if not fix_root_rot:
            end_rr = _interpolate_root_rot(end_rot_aa,
                                           default_rr_aa,
                                           end_ext_frames,
                                           reverse=True)

    # 合并数据
    new_root_trans = np.vstack([start_root_trans, root_trans, end_root_trans])
    if fix_root_rot:
        total_frames = start_ext_frames + input_data.shape[0] + end_ext_frames
        new_root_rot = np.tile(default_rr, (total_frames, 1))
    else:
        new_root_rot = np.vstack([start_rr, root_rot, end_rr])
    new_dof_pos = np.vstack([start_dof, dof_pos, end_dof])

    output_data = np.concatenate((new_root_trans, new_root_rot, new_dof_pos),
                                 axis=1)
    convert_pkl(output_data, output_pkl, contact_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--origin_file_name",
        type=str,
        help="Origin File name, shape (nframe, 36)",
        default="dance1_subject2.pkl",
    )
    parser.add_argument(
        "--fix_root_rot",
        type=bool,
        help="Fix default pose root rot. A DEBUG option, not a regular usage.",
        default=False,
    )
    parser.add_argument("--start", type=int, help="Start frame", default=0)
    parser.add_argument("--end", type=int, help="End frame", default=-1)
    parser.add_argument("--default_pose",
                        type=str,
                        help="Default Pose File name, shape (36,)")
    parser.add_argument("--start_inter_frame",
                        type=int,
                        help="Start Inter frame",
                        default=30)
    parser.add_argument("--end_inter_frame",
                        type=int,
                        help="End Inter frame",
                        default=30)
    args = parser.parse_args()

    # Load and clip data
    data = next(iter(joblib.load(args.origin_file_name).values()))
    print("Input DOF Shape: ", data["dof"].shape)

    end_frame = data["dof"].shape[0] if args.end == -1 else args.end
    print(f"Clip to {(args.start, end_frame)}")

    dof = data["dof"][args.start:end_frame]
    root_trans = data["root_trans_offset"][args.start:end_frame]
    root_rot = data["root_rot"][args.start:end_frame]
    contact_mask = (data["contact_mask"][args.start:end_frame]
                    if "contact_mask" in data else np.ones(
                        (end_frame - args.start, 2)) * 0.5)

    input_data = np.concatenate((root_trans, root_rot, dof), axis=1)

    # Load or use default pose
    if args.default_pose is not None:
        default_pose = np.load(args.default_pose)
    else:
        # yapf: disable
        default_pose = np.array([
            0.0, 0.0, 0.80,
            0.0, 0.0, 0.0, 1.0,
            -0.1,0.0,0.0, 0.3, -0.2, 0.0,
            -0.1,0.0,0.0, 0.3, -0.2, 0.0,
            0.0, 0.0, 0.0,
            0.2, 0.2, 0.0, 0.9, 0.0, 0.0, 0.0,
            0.2,-0.2, 0.0, 0.9, 0.0, 0.0, 0.0,
        ])
        # yapf: enable

    # Prepare contact mask for interpolation
    contact_in_interp = [0.5, 0.5]
    print("Contact in interpolation: ", contact_in_interp)

    contact_mask_start = np.tile([contact_in_interp],
                                 (args.start_inter_frame, 1))
    contact_mask_end = np.tile([contact_in_interp], (args.end_inter_frame, 1))
    contact_mask = np.concatenate(
        (contact_mask_start, contact_mask, contact_mask_end), axis=0)

    # Generate output filename
    output_filename = (
        str(
            Path(args.origin_file_name).parent /
            Path(args.origin_file_name).stem) +
        f"_inter{contact_in_interp[0]}_S{args.start}-{args.start_inter_frame}_E{end_frame}-{args.end_inter_frame}"
    )

    interpolate_motion(
        input_data=input_data,
        start_ext_frames=args.start_inter_frame,
        end_ext_frames=args.end_inter_frame,
        default_pose=default_pose,
        output_pkl=output_filename,
        contact_mask=contact_mask,
        fix_root_rot=args.fix_root_rot,
    )
