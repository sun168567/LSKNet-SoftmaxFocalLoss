import numpy as np
import cv2
from pathlib import Path

# 定义空间块大小
BLOCK_SIZE = 128

# 计算旋转矩形框的中心点
def get_center(rect):
    """
    计算旋转矩形框的中心点
    rect: ((x_center, y_center), (width, height), angle)
    """
    return rect[0]

# 判断矩形框A的中心点是否在矩形框B内
def is_center_inside(box_a, box_b):
    """
    判断矩形框A的中心点是否在矩形框B内部
    box_a 和 box_b 是旋转矩形框，格式为 ((x_center, y_center), (width, height), angle)
    """
    center_a = get_center(box_a)
    center_b = get_center(box_b)

    # 获取box_b的四个角坐标
    points_b = cv2.boxPoints(box_b)
    points_b = points_b.reshape((-1, 2))

    # 使用cv2.pointPolygonTest判断点是否在多边形内
    return cv2.pointPolygonTest(points_b, center_a, False) >= 0

# 读取并处理txt文件
def process_txt_file(txt_file_path):
    """
    读取txt文件，并根据空间块对矩形框进行分组。
    """
    results = {}
    original_data = []  # 用于保存原始数据
    
    # 读取txt文件
    with open(txt_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            label_id = int(parts[0])
            confidence = float(parts[1])
            coordinates = list(map(float, parts[2:]))
            
            # 保存原始数据
            original_data.append((label_id, confidence, coordinates))
            
            # 将xyxyxyxy格式转为OpenCV旋转矩形的形式
            points = np.array(coordinates).reshape((4, 2)).astype(np.float32)
            rect = cv2.minAreaRect(points)
            
            # 计算矩形框的中心点并映射到空间块
            center_x, center_y = get_center(rect)
            block_x = int(center_x // BLOCK_SIZE)
            block_y = int(center_y // BLOCK_SIZE)
            
            # 将框按空间块分组
            group_id = (block_x, block_y)
            if group_id not in results:
                results[group_id] = []
            results[group_id].append((label_id, confidence, rect, coordinates))

    print("按空间块分组完成")
    # 筛选重复的框，保留置信度高的框
    unique_results = []
    print(f"Total number of groups: {len(results)}")

    count = 0
    # 遍历每个空间块
    for group, boxes in results.items():
        count += 1  # 每次迭代时，计数器加1
        print(f"Processing group {count}/{len(results)}: {group}, have {len(boxes)} items")
        
        # 对组内框进行去重
        remaining_boxes = []
        for i in range(len(boxes)):
            rect1 = boxes[i][2]
            label_id1 = boxes[i][0]
            confidence1 = boxes[i][1]
            original_coords1 = boxes[i][3]
            remove = False
            
            # 对同一组的框进行两两比较
            for j in range(i + 1, len(boxes)):
                rect2 = boxes[j][2]
                label_id2 = boxes[j][0]
                confidence2 = boxes[j][1]
                original_coords2 = boxes[j][3]
                
                # 判断是否重合
                if is_center_inside(rect1, rect2):
                    if confidence1 > confidence2:
                        # 只保留置信度较高的框
                        boxes[j] = (label_id1, confidence1, rect1, original_coords1)
                    remove = True
                    break
            
            if not remove:
                remaining_boxes.append((label_id1, confidence1, rect1, original_coords1))
        
        # 将去重后的组结果加入到unique_results中
        unique_results.extend(remaining_boxes)
    
    # 返回去重后的结果（原始坐标）
    return unique_results

# 将去重后的结果写回文件
def save_filtered_results(txt_file_path, results):
    """
    将去重后的结果保存到txt文件
    """
    with open(txt_file_path, "w") as f:
        for label_id, confidence, rect, original_coords in results:
            # 使用原始数据，不经过cv2.minAreaRect的处理
            result_line = f"{label_id} {confidence} {' '.join(map(str, map(int, original_coords)))}\n"
            f.write(result_line)

# 主函数：处理指定路径下的所有txt文件
def filter_txt_files(input_dir, output_dir):
    """
    处理指定目录下的所有txt文件，去除重合框，保留置信度较高的框。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for txt_file in input_dir.glob("*.txt"):
        print(f"Processing {txt_file.name}...")
        
        # 处理当前txt文件
        filtered_results = process_txt_file(txt_file)
        
        # 保存去重后的结果
        output_txt_file = output_dir / txt_file.name
        save_filtered_results(output_txt_file, filtered_results)
        print(f"Filtered results saved to {output_txt_file}")

# 使用示例
input_dir = "./outputs"  # 输入txt文件的目录
output_dir = "./filtered_outputs"  # 输出去重后的txt文件目录
filter_txt_files(input_dir, output_dir)
