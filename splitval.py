import os
import shutil
import re

# 定义路径
base_dir = './data/split_ss_fair1m2_0'
val_images_dir = os.path.join(base_dir, 'val/images')
val_annfiles_dir = os.path.join(base_dir, 'val/annfiles')
test_images_dir = os.path.join(base_dir, 'test/images')
test_annfiles_dir = os.path.join(base_dir, 'test/annfiles')

# 创建目标文件夹
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_annfiles_dir, exist_ok=True)

# 提取排序键的函数：获取前缀ID作为整数
def get_id(filename):
    match = re.match(r"(\d+)__\d+__\d+___\d+\.png", filename)
    return int(match.group(1)) if match else float('inf')

# 获取所有图像文件名并排序
image_files = sorted([f for f in os.listdir(val_images_dir) if f.endswith('.png')], key=get_id)

# 计算一半的数量
half_len = len(image_files) // 2
selected_images = image_files[:half_len]

# 复制图片和对应标签
for img_file in selected_images:
    base_name = os.path.splitext(img_file)[0]
    label_file = base_name + '.txt'

    src_img = os.path.join(val_images_dir, img_file)
    src_label = os.path.join(val_annfiles_dir, label_file)
    dst_img = os.path.join(test_images_dir, img_file)
    dst_label = os.path.join(test_annfiles_dir, label_file)

    # 移动图像和标签
    if os.path.exists(src_img):
        shutil.move(src_img, dst_img)
    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)
