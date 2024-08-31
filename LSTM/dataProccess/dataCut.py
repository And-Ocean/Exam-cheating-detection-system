import os

def process_txt_files(directory):
    # 获取目录下所有的txt文件
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    for file in txt_files:
        file_path = os.path.join(directory, file)
        
        # 读取文件内容
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 处理每行数据
        new_lines = []
        for line in lines:
            data = line.strip().split(' ')
            if len(data) == 57:
                new_line = ' '.join(data[:-1])
                new_lines.append(new_line)
            else:
                new_lines.append(line.strip())
        
        # 写回文件
        with open(file_path, 'w') as f:
            for new_line in new_lines:
                f.write(new_line + '\n')

# 指定文件夹路径
directory = 'E:/video2'

# 处理文件夹下的所有txt文件
process_txt_files(directory)
