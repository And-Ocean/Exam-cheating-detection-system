import os

def rename_files_sequentially(data_dir):
    # 获取所有文件名，并按章节和页码排序
    filenames = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')],
                    key=lambda x: (int(x.split('_')[0]),
                    int(x.split('_')[1].split('.')[0])))
    
    # 重新命名文件
    for idx, filename in enumerate(filenames):
        # 计算新的文件名
        new_filename = f'1_{idx + 1}.txt'
        
        # 构建文件路径
        old_filepath = os.path.join(data_dir, filename)
        new_filepath = os.path.join(data_dir, new_filename)
        
        # 重命名文件
        os.rename(old_filepath, new_filepath)
        print(f'Renamed {filename} to {new_filename}')

# 文件目录路径
data_dir = 'E:/video1'  # 替换为你的TXT文件目录

rename_files_sequentially(data_dir)
