import argparse
import os
import os.path as osp
import pdb
import sys
import time

# import pinocchio as pin

sys.path.append(os.getcwd())

# from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
import math
from collections import defaultdict
from copy import deepcopy
from typing import List, Union

import hydra
import joblib
import mujoco
import mujoco.viewer
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as sRot


def motion2q(motion):
    qpos = np.concatenate(
        [
            motion["root_trans_offset"],
            motion["root_rot"][:, [3, 0, 1, 2]],
            motion["dof"],
        ],
        axis=1,
    )
    if not "root_lin_vel" in motion:
        # nv = qpos.shape[1]-1
        # current_qvel = np.zeros(nv)
        # mujoco.mj_differentiatePos(model, current_qvel, rate.dt,
        #                             prev_qpos, current_qpos)
        qvel = np.zeros((qpos.shape[0], qpos.shape[1] - 1))
    else:
        qvel = np.concatenate(
            [
                motion["root_lin_vel"].reshape(-1, 3),
                motion["root_ang_vel"].reshape(-1, 3),
                motion["dof_vel"],
            ],
            axis=1,
        )
    return qpos, qvel


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        point1[0],
        point1[1],
        point1[2],
        point2[0],
        point2[1],
        point2[2],
    )


def key_call_back(keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind, motion_data_keys, contact_mask, curr_time, resave
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    elif keycode == 256 or chr(keycode) == "Q":
        print("Esc")
        os._exit(0)
    elif chr(keycode) == "L":
        speed = speed * 1.5
        print("Speed: ", speed)
    elif chr(keycode) == "K":
        speed = speed / 1.5
        print("Speed: ", speed)
    elif chr(keycode) == "J":
        print("Toggle Rewind: ", not rewind)
        rewind = not rewind
    elif keycode == 262:  # (Right)
        time_step += dt
    elif keycode == 263:  # (Left)
        time_step -= dt
    elif chr(keycode) == "Q":
        print("Modify left foot contact!!!")
        contact_mask[curr_time][0] = 1.0 - contact_mask[curr_time][0]
        resave = True
    elif chr(keycode) == "E":
        print("Modify right foot contact!!!")
        contact_mask[curr_time][1] = 1.0 - contact_mask[curr_time][1]
        resave = True
    elif chr(keycode) == "O":
        print("next motion")
        motion_id = (motion_id + 1) % len(motion_data_keys)
        time_step = 0
    elif chr(keycode) == "P":
        print("prev motion")
        motion_id = (motion_id - 1) % len(motion_data_keys)
        time_step = 0
    elif chr(keycode) == "M":
        try:
            new_id = int(
                input(
                    f"\nEnter new motion_id (0 ~ {len(motion_data_keys) - 1}): "
                ))
            if 0 <= new_id < len(motion_data_keys):
                motion_id = new_id
                time_step = 0
                print(f"Switched to motion_id {motion_id}")
            else:
                print("Invalid motion_id!")
        except Exception as e:
            print("Invalid input!", e)
    else:
        print("not mapped", chr(keycode), keycode)


def get_com(model, data):
    # Get CoM of entire model
    com = np.zeros(3)
    total_mass = 0.0

    for i in range(model.nbody):
        mass = model.body_mass[i]
        xpos = data.xipos[i]  # position of body CoM in world frame
        com += mass * xpos
        total_mass += mass

    com /= total_mass
    return com


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind, motion_data_keys, contact_mask, curr_time, resave
    (
        curr_start,
        num_motions,
        motion_id,
        motion_acc,
        time_step,
        dt,
        speed,
        paused,
        rewind,
    ) = 0, 1, 0, set(), 0, 1 / 30, 1.0, False, False
    # if 'dt' in cfg:
    #     dt = cfg.dt
    motion_file = cfg.motion_file
    motion_data = joblib.load(motion_file)
    motion_data_keys = list(motion_data.keys())
    curr_motion_key = motion_data_keys[motion_id]
    curr_motion = motion_data[curr_motion_key]
    print(motion_file)

    speed = 1.0 if "speed" not in cfg else cfg.speed
    hang = False if "hang" not in cfg else cfg.hang
    if hang:
        curr_motion["root_trans_offset"][:] = np.array([0, 0, 0.8])
    if "fps" in curr_motion:
        dt = 1.0 / curr_motion["fps"]
    elif "dt" in cfg:
        dt = cfg.dt

    print("Motion file: ", motion_file)
    print(f"Switched to motion: {curr_motion_key}")
    print("Motion length: ", motion_data[motion_data_keys[0]]["dof"].shape[0],
          "frames")
    print(
        "Max motion length:",
        max(motion_data[k]["dof"].shape[0] for k in motion_data_keys),
    )
    print(
        "Avg motion length:",
        sum(motion_data[k]["dof"].shape[0]
            for k in motion_data_keys) / len(motion_data_keys),
    )

    print("Speed: ", speed)
    print()

    curr_time = 0
    resave = False

    humanoid_xml = "./assets/unitree_g1/g1_mocap_29dof.xml"
    print(humanoid_xml)

    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt

    qpos_list, qvel_list = motion2q(curr_motion)

    print(
        "Init Pose: ",
        (np.array(
            np.concatenate([
                curr_motion["root_trans_offset"][0],
                curr_motion["root_rot"][0][[3, 0, 1, 2]],
                curr_motion["dof"][0],
            ]),
            dtype=np.float32,
        )).__repr__(),
    )

    with mujoco.viewer.launch_passive(mj_model,
                                      mj_data,
                                      key_callback=key_call_back) as viewer:
        viewer.cam.lookat[:] = np.array([0, 0, 0.7])
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -30  # 负值表示从上往下看viewer

        prev_motion_id = motion_id
        # breakpoint()
        while viewer.is_running():
            if motion_id != prev_motion_id:
                curr_motion_key = motion_data_keys[motion_id]
                curr_motion = motion_data[curr_motion_key]
                print(f"Switched to motion: {curr_motion_key}")
                print(
                    "Motion length: ",
                    motion_data[curr_motion_key]["dof"].shape[0],
                    "frames",
                )

                if "fps" in curr_motion:
                    dt = 1.0 / curr_motion["fps"]
                elif "dt" in cfg:
                    dt = cfg.dt

                mj_model.opt.timestep = dt
                qpos_list, qvel_list = motion2q(curr_motion)

                contact_mask = curr_motion.get("contact_mask", None)
                time_step = 0
                prev_motion_id = motion_id
            step_start = time.time()
            if time_step >= curr_motion["dof"].shape[0] * dt:
                time_step -= curr_motion["dof"].shape[0] * dt
            curr_time = round(time_step / dt) % curr_motion["dof"].shape[0]

            mj_data.qpos = qpos_list[curr_time]
            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt * (1 if not rewind else -1) * speed

            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() -
                                                            step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            print("Frame ID: ",
                  curr_time,
                  "\t | Times ",
                  f"{time_step:4f}",
                  end="\r\b")

    if resave:
        motion_data[curr_motion_key]["contact_mask"] = contact_mask
        motion_file = motion_file.split(".")[0] + "_edit_cont.pkl"
        print(motion_file)
        joblib.dump(motion_data, motion_file)


if __name__ == "__main__":
    main()
