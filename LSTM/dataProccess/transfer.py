import os

# 读取文件内容到数组，并修改内容
def read_modify_write(file_path,new_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append([float(a) for a in line.split()])

        for line in lines:
            x,y,w,h = line[1],line[2],line[3],line[4]
            x -= w/2
            y -= h/2
            for i in range(5,len(line)-1,3):
                line[i] = (line[i] - x) * w
                line[i+1] = (line[i+1] -y) * h
    with open(new_path, 'w', encoding='utf-8') as file:
        for line in lines:
            str = ''
            for i in line:
                if i == 0:
                    str += '0 '
                else:
                    str += f"{i:.6f} "
            file.write(str + '\n')
    print(new_path+'修改成功')

# 遍历文件夹中的所有txt文件，并对每个文件执行操作
def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, 'new_'+filename)
            read_modify_write(file_path,new_path)

process_folder(r"F:\cheating_detection\Exam-cheating-detection-system-main\runs\pose\predict\lables")