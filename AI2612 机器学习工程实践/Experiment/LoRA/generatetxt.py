# import os
# from pathlib import Path

# def generate_txt_files(source_dir, prefix):
#     # 将源文件夹路径转换为 Path 对象
#     source_dir = Path(source_dir)
    
#     # 确保源文件夹存在
#     if not source_dir.exists():
#         raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")
    
#     # 递归遍历文件夹中的所有文件
#     for file_path in source_dir.rglob('*'):
#         if file_path.is_file():
#             # 生成目标文本文件的路径
#             txt_file_path = file_path.with_suffix('.txt')
            
#             # 生成文件路径内容，舍去固定前缀
#             content = str(file_path.parent).replace(prefix, '', 1)
#             content = ','.join(content.split('/'))
#             # 写入内容到文本文件
#             with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
#                 txt_file.write(content)
#             print(f"Generated {txt_file_path} with content: {content}")

# # 示例用法
# path="./data/comic"
# path = Path(path)
# for source_dir in path.rglob('*'):
    
#     # source_dir = './data/graphic_design/poster_no_edge'
#     prefix = 'data/'
#     generate_txt_files(source_dir, prefix)

import os
import json

root_dir = "./data/comic"
metadata_file = os.path.join(root_dir, "metadata.jsonl")

# 定义支持的图片格式
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]

# 初始化 metadata 列表
metadata = []

# 遍历目录中的所有文件
for filename in os.listdir(root_dir):
    # 检查文件扩展名是否在支持的图片格式列表中
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        # 获取对应的 TXT 文件名
        base_name = os.path.splitext(filename)[0]
        txt_filename = f"{base_name}.txt"
        txt_path = os.path.join(root_dir, txt_filename)
        
        # 检查对应的 TXT 文件是否存在
        if os.path.exists(txt_path):
            # 读取 TXT 文件中的风格描述
            with open(txt_path, 'r', encoding='utf-8') as txt_file:
                style = txt_file.read().strip()
            
            # 生成 metadata.jsonl 文件中的内容
            metadata_entry = {
                "file_name": filename,
                "text": f"An image in {style} style"
            }
            metadata.append(metadata_entry)
        else:
            print(f"Warning: No corresponding TXT file found for {filename}")

# 写入 metadata.jsonl 文件
with open(metadata_file, 'w', encoding='utf-8') as jsonl_file:
    for entry in metadata:
        jsonl_file.write(json.dumps(entry) + '\n')

print(f"Generated {metadata_file} with {len(metadata)} entries.")