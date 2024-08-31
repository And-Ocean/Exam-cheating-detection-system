import cv2
import os

# 定义根目录路径
root_dir = "src/video_train"

# 支持的视频格式列表
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

# 遍历根目录下的所有子目录
for subdir, _, files in os.walk(root_dir):
    # 获取当前子目录中的所有视频文件
    video_files = [f for f in files if os.path.splitext(f)[1].lower() in video_extensions]

    # 如果当前子目录中没有找到视频文件，继续遍历下一个子目录
    if not video_files:
        continue

    # 定义步长
    step = 15

    # 遍历当前子目录中的每个视频文件
    for video_name in video_files:
        video_path = os.path.join(subdir, video_name)
        output_file = os.path.join(subdir, f"{os.path.splitext(video_name)[0]}_labels.txt")
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Cannot open the video file {video_name}.")
            continue

        # 设置窗口的名称和大小
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Frame", 640, 480)  # 调整窗口大小

        # 初始化一个列表来保存用户的输入
        labels = []

        frame_number = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while frame_number < total_frames:
            # 读取当前帧
            ret, frame = cap.read()

            if not ret:
                print(f"End of video {video_name}.")
                break

            # 调整帧的大小以适应窗口
            resized_frame = cv2.resize(frame, (640, 480))

            # 显示调整大小后的当前帧
            cv2.imshow("Frame", resized_frame)

            # 提示用户输入
            print(f"Video: {video_name}, Frame {frame_number}: Enter 0 or 1:")
            key = input()
            while key not in ["0", "1"]:
                print("Invalid input. Please enter 0 or 1.")
                key = input()

            # 检查剩余帧数是否大于或等于步长
            if frame_number + step < total_frames:
                # 剩余帧数大于步长，添加步长个标签
                labels.extend([key] * step)
                frame_number += step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            else:
                # 剩余帧数小于步长，逐帧处理
                labels.append(key)
                frame_number += 1

            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 关闭视频窗口
        cap.release()
        cv2.destroyAllWindows()

        # 将用户输入的值写入txt文件
        with open(output_file, "w") as f:
            f.write(" ".join(labels))

        print(f"Labels saved to {output_file}")
