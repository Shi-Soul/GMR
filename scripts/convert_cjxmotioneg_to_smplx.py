import os
import numpy as np
import joblib
from tqdm import tqdm  # 进度条库
import shutil
from scipy.spatial.transform import Rotation as sRot
import argparse


def process_pkl_file(input_path, output_path):
    """
    处理单个pkl文件的示例函数
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :return: 处理是否成功
    """
    try:
        data = joblib.load(input_path)

        betas = data['beta'][0]
        trans = data['cam'][:, 0]
        gender = 'neutral'
        poses = data['pose'][:, 0]
        mocap_framerate = 30

        processed_data = {}

        if betas.shape[0] == 16:
            processed_data['betas'] = betas
        elif betas.shape[0] == 10:
            processed_data['betas'] = np.concatenate([betas, np.zeros(6)])
        else:
            raise ValueError("Betas shape is not supported: {}".format(
                betas.shape))

            # y-up to z-up
        transform = sRot.from_euler('xyz',
                                    np.array([np.pi / 2, 0, np.pi]),
                                    degrees=False)
        root_rot = (
            transform *
            sRot.from_euler('xyz', poses[:, :3], degrees=False)).as_rotvec()
        trans = trans @ (transform.as_matrix().T)

        processed_data['trans'] = trans
        processed_data['gender'] = gender
        processed_data['root_orient'] = root_rot
        processed_data['pose_body'] = poses[:, 3:66]
        processed_data['mocap_frame_rate'] = mocap_framerate

        output_path = output_path.replace('.pkl', '')
        # 3. 保存处理后的数据
        np.savez(output_path, **processed_data)
        return True
    except Exception as e:
        print(f"\n处理文件 {input_path} 时出错: {str(e)}")
        return False


def process_directory(source_dir, target_dir, file_extension='.pkl'):
    """
    处理整个目录及其子目录中的npz文件
    :param source_dir: 源目录路径
    :param target_dir: 目标目录路径
    :param file_extension: 要处理的文件扩展名
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    pkl_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(file_extension):
                full_path = os.path.join(root, file)  # 确保目标目录存在
                pkl_files.append(full_path)

    total_files = len(pkl_files)
    print(f"找到 {total_files} 个 {file_extension} 文件需要处理")

    # 使用tqdm显示进度条
    for file_path in tqdm(pkl_files, desc="处理进度"):
        # 计算相对路径
        relative_path = os.path.relpath(file_path, source_dir)
        output_path = os.path.join(target_dir, relative_path)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 处理文件
        process_pkl_file(file_path, output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        help="Folder containing BVH motion files to load.",
        required=True,
        type=str,
    )

    parser.add_argument("--tgt",
                        help="Folder to save the retargeted motion files.",
                        default="../../motion_data/LAFAN1_g1_gmr")

    args = parser.parse_args()

    source_directory = args.src
    target_directory = args.tgt

    process_directory(source_directory, target_directory)
