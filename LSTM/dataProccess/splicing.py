import os
import cv2
import numpy as np
import re

def clean_text(content):
    # 删除所有非0、1或空格的字符
    cleaned_content = re.sub(r'[^01\s]', '', content)
    return cleaned_content

def concatenate_videos_and_txt(folder_path):
    # 获取文件夹名称
    folder_name = os.path.basename(folder_path)
    
    # 获取所有视频文件和txt文件
    video_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mp4')])
    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])

    # 初始化视频输出
    video_writer = None
    width, height = None, None
    fps = None

    # 拼接txt文件
    with open(os.path.join(folder_path, f'{folder_name}.txt'), 'w', encoding='utf-8') as outfile:
        for txt_file in txt_files:
            with open(os.path.join(folder_path, txt_file), 'r', encoding='utf-8') as infile:
                content = infile.read()
                cleaned_content = clean_text(content)
                outfile.write(cleaned_content)
                outfile.write(' ')  # 添加空格

    # 拼接视频文件
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_file}")
            continue

        # 获取视频的属性
        if video_writer is None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # 初始化视频写入器
            output_video_path = os.path.join(folder_path, f'{folder_name}.mp4')
            video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_writer.write(frame)

        cap.release()

    if video_writer:
        video_writer.release()

    print(f"拼接完成: {folder_name}.mp4 和 {folder_name}.txt")

def process_all_subdirectories(root_dir):
    # 遍历根目录下的所有子目录
    for subdir, _, _ in os.walk(root_dir):
        # 跳过根目录自身
        if subdir == root_dir:
            continue
        
        # 调用拼接函数处理每个子目录
        concatenate_videos_and_txt(subdir)

# 使用示例
root_dir = 'src/video_train'
process_all_subdirectories(root_dir)
