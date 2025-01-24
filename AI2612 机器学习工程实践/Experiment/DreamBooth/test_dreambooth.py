from diffusers import StableDiffusionPipeline
import torch
from diffusers import DDIMScheduler
import os

# 模型路径和提示词
model_path = "/root/autodl-tmp/dreambooth-for-diffusion/final-model-1"
prompt = "221 style house"

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                               torch_dtype=torch.float16,
                                               scheduler=DDIMScheduler(
                                                   beta_start=0.00085,
                                                   beta_end=0.012,
                                                   beta_schedule='scaled_linear',
                                                   clip_sample=False,
                                                   set_alpha_to_one=True),
                                                   safety_checker=None
                                                   )

# 将模型迁移到 GPU
pipe = pipe.to("cuda")

# 生成图像
images = pipe(prompt, num_inference_steps=30, num_images_per_prompt=3).images

# 保存生成的图像
save_path = "./final-model-test"  # 替换成你想要的路径
if not os.path.exists(save_path):
    os.makedirs(save_path)  # 如果目录不存在，创建目录
for i, image in enumerate(images):
    image.save(os.path.join(save_path, f"{prompt}-{i}.png"))
