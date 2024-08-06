import os

def get_abnormal_ranges():
    abnormal_ranges = []
    while True:
        start = input("输入异常行为起始帧（或输入'q'结束）：")
        if start.lower() == 'q':
            break
        end = input("输入异常行为结束帧：")
        abnormal_ranges.append((int(start), int(end)))
    return abnormal_ranges

def is_abnormal_frame(frame_number, abnormal_ranges):
    for start, end in abnormal_ranges:
        if start <= frame_number <= end:
            return True, frame_number - start + 1
    return False, None

def rename_files(data_dir, abnormal_ranges):
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt') and filename.startswith('test4_'):
            # 提取帧序号
            frame_number_str = filename.split('_')[1].split('.')[0]
            frame_number = int(frame_number_str)
            
            # 根据帧序号确定行为类型
            is_abnormal, abnormal_frame_number = is_abnormal_frame(frame_number, abnormal_ranges)
            if is_abnormal:
                # 异常行为帧
                new_filename = f'frame_{frame_number_str.zfill(4)}_abnormal_{str(abnormal_frame_number).zfill(4)}.txt'
            else:
                # 正常行为帧
                new_filename = f'frame_{frame_number_str.zfill(4)}_normal.txt'
            
            # 重命名文件
            old_filepath = os.path.join(data_dir, filename)
            new_filepath = os.path.join(data_dir, new_filename)
            os.rename(old_filepath, new_filepath)
            print(f'Renamed {filename} to {new_filename}')

# 获取异常行为范围
abnormal_ranges = get_abnormal_ranges()

# 文件目录路径
data_dir = 'runs/pose/predict4/labels'

# 重命名文件
rename_files(data_dir, abnormal_ranges)
