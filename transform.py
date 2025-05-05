import os

# 标签列表
CLASSES = (
    'Boeing737', 'Boeing777', 'Boeing747', 'Boeing787', 'A321',
    'A220', 'A330', 'A350', 'C919', 'ARJ21', 'other-airplane', 
    'Passenger_Ship', 'Motorboat', 'Fishing_Boat', 'Tugboat', 'Engineering_Ship', 
    'Liquid_Cargo_Ship', 'Dry_Cargo_Ship', 'Warship', 'other-ship', 'Small_Car', 'Bus', 'Cargo_Truck', 
    'Dump_Truck', 'Van', 'Trailer', 'Tractor', 'Truck_Tractor', 'Excavator', 'other-vehicle',
    'Baseball_Field', 'Basketball_Court', 'Football_Field', 'Tennis_Court', 'Roundabout', 'Intersection', 'Bridge'
)

# 工作目录
work_dir = './work_dirs'

# 合并输出文件路径
output_file = 'merged_output.txt'

# 打开输出文件
with open(output_file, 'w') as out_file:
    # 遍历所有txt文件
    for i, cls in enumerate(CLASSES):
        # 构建对应文件路径
        file_name = f'Task1_{cls}.txt'
        file_path = os.path.join(work_dir, file_name)

        # 检查文件是否存在
        if os.path.exists(file_path):
            # 读取每个txt文件
            with open(file_path, 'r') as in_file:
                for line in in_file:
                    # 拆分每行数据
                    parts = line.split()
                    image_id = int(parts[0])  # image_id为整数
                    confidence = float(parts[1])  # confidence为浮动类型
                    # 获取旋转矩形框四个点坐标
                    points = [int(float(coord)) for coord in parts[2:]]  # 坐标转换为整数

                    # 写入转换后的数据到输出文件
                    out_file.write(f'{i} {confidence:.6f} ' + ' '.join(map(str, points)) + '\n')

print(f"合并后的文件已保存至: {output_file}")
