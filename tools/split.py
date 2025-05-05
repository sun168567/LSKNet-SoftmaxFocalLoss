import os
import numpy as np
from PIL import Image
import rasterio

# 配置参数
input_dir = 'F:/FAIR1M2.0/ultralytics-main/datasets/guangzhou_port'  # 原始遥感图像所在目录
output_dir = 'F:/FAIR1M2.0/ultralytics-main/datasets/guangzhou_port_split/images'  # 分割图像输出目录
block_size = 1024  # 滑动块大小
stride = 512  # 滑动块移动步长

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录中的所有tif文件
for filename in os.listdir(input_dir):
    if filename.endswith('.tif'):
        image_path = os.path.join(input_dir, filename)
        
        # 使用 rasterio 打开遥感图像
        with rasterio.open(image_path) as src:
            img_width = src.width
            img_height = src.height
            num_bands = src.count  # 获取图像的波段数（彩色图像通常是3）
            
            # 读取图像的元数据来确定如何切割图像
            for y in range(0, img_height - block_size + 1, stride):
                for x in range(0, img_width - block_size + 1, stride):
                    # 计算当前滑动块的窗口
                    window = rasterio.windows.Window(x, y, block_size, block_size)
                    
                    # 读取滑动块的数据
                    block = src.read([1, 2, 3], window=window)  # 读取三个波段的颜色数据 (假设是RGB)
                    
                    # 将读取到的图像数据转换为3通道彩色图像
                    block_image = np.moveaxis(block, 0, -1)  # 将波段维度移到最后形成 (height, width, channels)
                    
                    # 如果图像块小于1024x1024，填充黑色
                    if block_image.shape[0] < block_size or block_image.shape[1] < block_size:
                        block_image = np.pad(block_image, ((0, block_size - block_image.shape[0]),
                                                            (0, block_size - block_image.shape[1]),
                                                            (0, 0)), mode='constant', constant_values=0)
                    
                    # 转换为PIL图像对象并保存为PNG文件
                    block_image_pil = Image.fromarray(block_image)
                    output_filename = f"{filename.split('.')[0]}_{x // stride}_{y // stride}.png"
                    block_image_pil.save(os.path.join(output_dir, output_filename))

            # 处理右边界和下边界的滑动块（可能是填充黑色）
            for y in range(0, img_height - block_size + 1, stride):
                for x in range(img_width - block_size, img_width, stride):
                    window = rasterio.windows.Window(x, y, block_size, block_size)
                    block = src.read([1, 2, 3], window=window)
                    block_image = np.moveaxis(block, 0, -1)
                    if block_image.shape[1] < block_size:
                        block_image = np.pad(block_image, ((0, 0), (0, block_size - block_image.shape[1]), (0, 0)),
                                             mode='constant', constant_values=0)
                    block_image_pil = Image.fromarray(block_image)
                    output_filename = f"{"0".split('.')[0]}_{x // stride}_{y // stride}.png"
                    block_image_pil.save(os.path.join(output_dir, output_filename))

            for x in range(0, img_width - block_size + 1, stride):
                for y in range(img_height - block_size, img_height, stride):
                    window = rasterio.windows.Window(x, y, block_size, block_size)
                    block = src.read([1, 2, 3], window=window)
                    block_image = np.moveaxis(block, 0, -1)
                    if block_image.shape[0] < block_size:
                        block_image = np.pad(block_image, ((0, block_size - block_image.shape[0]), (0, 0), (0, 0)),
                                             mode='constant', constant_values=0)
                    block_image_pil = Image.fromarray(block_image)
                    output_filename = f"{"0".split('.')[0]}_{x // stride}_{y // stride}.png"
                    block_image_pil.save(os.path.join(output_dir, output_filename))

            # 处理右下角的部分，可能需要同时填充边界
            for y in range(img_height - block_size, img_height, stride):
                for x in range(img_width - block_size, img_width, stride):
                    window = rasterio.windows.Window(x, y, block_size, block_size)
                    block = src.read([1, 2, 3], window=window)
                    block_image = np.moveaxis(block, 0, -1)
                    if block_image.shape[0] < block_size or block_image.shape[1] < block_size:
                        block_image = np.pad(block_image, ((0, block_size - block_image.shape[0]),
                                                            (0, block_size - block_image.shape[1]),
                                                            (0, 0)), mode='constant', constant_values=0)
                    block_image_pil = Image.fromarray(block_image)
                    output_filename = f"{"0".split('.')[0]}_{x // stride}_{y // stride}.png"
                    block_image_pil.save(os.path.join(output_dir, output_filename))
