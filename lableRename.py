import os

def rename_files(data_dir, abnormal_start, abnormal_end):
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt') and filename.startswith('test1_'):
            # 提取帧序号
            frame_number_str = filename.split('_')[1].split('.')[0]
            frame_number = int(frame_number_str)
            
            # 根据帧序号确定行为类型
            if abnormal_start <= frame_number <= abnormal_end:
                # 异常行为帧
                abnormal_frame_number = frame_number - abnormal_start + 1
                new_filename = f'frame_{frame_number_str.zfill(4)}_abnormal_{str(abnormal_frame_number).zfill(4)}.txt'
            else:
                # 正常行为帧
                new_filename = f'frame_{frame_number_str.zfill(4)}_normal.txt'
            
            # 重命名文件
            old_filepath = os.path.join(data_dir, filename)
            new_filepath = os.path.join(data_dir, new_filename)
            os.rename(old_filepath, new_filepath)
            print(f'Renamed {filename} to {new_filename}')

# 文件目录路径
data_dir = 'runs\pose\predict2\labels'  # 替换为你的TXT文件目录

# 异常行为帧范围
abnormal_start = 240
abnormal_end = 360

rename_files(data_dir, abnormal_start, abnormal_end)
