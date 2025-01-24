import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    return img  # 返回图像数据

images_folder = 'images_GID_visualize'
output_folder_masks = 'outputs_masks_yaogan'
output_folder = 'output_GID_visualize_SAM'
sys.path.append("..")

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

for filename in os.listdir(images_folder):
    image_path = os.path.join(images_folder, filename)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_generator = SamAutomaticMaskGenerator(sam)

    masks = mask_generator.generate(image)
    
    # 构造保存masks的文件夹路径
    masks_folder = os.path.join(output_folder_masks, os.path.splitext(filename)[0])
    if not os.path.exists(masks_folder):
        os.makedirs(masks_folder)

    # 为每个mask创建一个唯一的文件名并保存
    for i, mask in enumerate(masks):
        mask_filename = f'mask_{i}.npy'  # 创建mask的文件名
        mask_output_path = os.path.join(masks_folder, mask_filename)
        np.save(mask_output_path, mask)  # 保存每个mask为单独的.npy文件
    
    # 创建一个空集合来存储出现过的数字
    unique_numbers = set()

    # 遍历每个mask
    for mask in masks:
        #print(mask)
        # 将当前mask展平成一维数组
        flat_mask = mask['segmentation'].flatten()
        # 更新集合，添加新的数字
        unique_numbers.update(flat_mask)
    
    # 为每个mask创建一个唯一的文件名并保存
    for i, mask in enumerate(masks):
        mask_255 = np.where(mask['segmentation'] == 0, 0, 255).astype(np.uint8)
        mask_filename = f'mask_{i}.png'  # 创建mask的文件名
        mask_output_path = os.path.join(masks_folder, mask_filename)
        cv2.imwrite(mask_output_path, mask_255)  # 保存每个mask为单独的PNG文件

    output_image = show_anns(masks)
    if output_image.dtype != np.uint8:
        output_image = (output_image * 255).astype(np.uint8)
    # 创建一个新的数组，形状为 [1024, 1024, 4]
    expanded_image = np.zeros((height, width, 4), dtype=np.uint8)

    # 将原图像的数据复制到新数组的前三个通道
    expanded_image[:, :, 0:3] = image

    # 假设原图像的第四通道实际上是第三通道（因为数组索引从0开始）
    expanded_image[:, :, 3] = 150
    
    # 读取两张图像的RGB通道和A通道（透明度）
    rgb1, alpha1 = expanded_image[:, :, :3], expanded_image[:, :, 3] / 255.0
    rgb2, alpha2 = output_image[:, :, :3], output_image[:, :, 3] / 255.0

    # 计算混合权重，使用两张图像的透明度值
    # 如果两张图像的透明度值都很低，则结果会接近0，如果都很高，则结果会接近1
    # 这里使用简单的加权平均，你可以根据需要调整混合策略
    combined_alpha = alpha1 + alpha2
    combined_rgb = np. zeros((height, width,3), dtype=np.float32)
    # 计算混合后的RGB值
    # 这里使用两张图像的透明度值作为权重，对它们的颜色值进行加权平均
    for i in range(3):
        combined_rgb[:,:,i] = rgb1[:,:,i] * (alpha1 / combined_alpha) + rgb2[:,:,i] * (alpha2 * (1 - alpha1) / combined_alpha)

    # 将混合后的RGB值和透明度值组合成最终的图像
    output_image = np.dstack((combined_rgb, combined_alpha * 255)).astype(np.uint8)
    output_image = Image.fromarray(output_image)
    output_path = os.path.join(output_folder, filename)
    output_image.save(output_path)

