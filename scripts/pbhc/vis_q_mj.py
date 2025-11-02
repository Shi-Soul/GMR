import os
import sys
import time

sys.path.append(os.getcwd())

import hydra
import joblib
import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import DictConfig


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


def key_call_back(keycode):
    global motion_id, time_step, dt, speed, paused, rewind, motion_data_keys, contact_mask, curr_time, resave
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    elif keycode == 256:  # ESC
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
    elif keycode == 262:  # Right arrow
        time_step += dt
    elif keycode == 263:  # Left arrow
        time_step -= dt
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


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    global motion_id, time_step, dt, speed, paused, rewind, motion_data_keys, contact_mask, curr_time, resave
    motion_id = 0
    time_step = 0
    dt = 1 / 30
    paused = False
    rewind = False

    motion_file = cfg.motion_file
    motion_data = joblib.load(motion_file)
    motion_data_keys = list(motion_data.keys())
    curr_motion_key = motion_data_keys[motion_id]
    curr_motion = motion_data[curr_motion_key]

    speed = cfg.get("speed", 1.0)
    hang = cfg.get("hang", False)
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
