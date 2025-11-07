import argparse
import os
import shutil

import numpy as np
from tqdm import tqdm  # 进度条库


def process_npz_file(input_path, output_path):
    """
    处理单个npz文件的示例函数
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :return: 处理是否成功
    """
    try:
        # 1. 加载npz文件
        data = np.load(input_path)

        betas = data['betas']
        trans = data['trans']
        gender = data['gender']
        poses = data['poses']
        mocap_framerate = data['mocap_framerate']

        processed_data = {}

        if betas.shape[0] == 16:
            processed_data['betas'] = betas
        elif betas.shape[0] == 10:
            processed_data['betas'] = np.concatenate([betas, np.zeros(6)])
        else:
            raise ValueError("Betas shape is not supported: {}".format(
                betas.shape))

        processed_data['trans'] = trans
        processed_data['gender'] = gender
        processed_data['root_orient'] = poses[:, :3]
        processed_data['pose_body'] = poses[:, 3:66]
        processed_data['mocap_frame_rate'] = mocap_framerate

        # 3. 保存处理后的数据
        np.savez(output_path, **processed_data)
        return True
    except Exception as e:
        print(f"\n处理文件 {input_path} 时出错: {str(e)}")
        return False


def process_directory(source_dir, target_dir):
    """
    处理整个目录及其子目录中的npz文件
    :param source_dir: 源目录路径
    :param target_dir: 目标目录路径
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 收集所有npz文件
    npz_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.npz'):
                full_path = os.path.join(root, file)
                npz_files.append(full_path)

    total_files = len(npz_files)
    print(f"找到 {total_files} 个 .npz 文件需要处理")

    # 使用tqdm显示进度条
    for file_path in tqdm(npz_files, desc="处理进度"):
        # 计算相对路径
        relative_path = os.path.relpath(file_path, source_dir)
        output_path = os.path.join(target_dir, relative_path)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 处理文件
        process_npz_file(file_path, output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TeleAI npz files to GMR npz format.")
    parser.add_argument(
        "--source",
        required=True,
        help="Source directory containing input npz files.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target directory to write processed npz files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_directory(
        source_dir=args.source,
        target_dir=args.target,
    )


if __name__ == "__main__":
    main()
