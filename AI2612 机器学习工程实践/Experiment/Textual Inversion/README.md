### 关于textual inversion微调代码的使用

主要依赖安装：可在notebook里执行以下代码inversion

```python
!pip install torch torchvision transformers diffusers tqdm
```

基础模型：`models--CompVis--stable-diffusion-v1-4` 默认从本地读取，如果想使用其他模型，可由`model_dir`修改模型路径；也可选择直接从huggingface下载

微调数据集准备：设置 `image_folder` 为数据目录

checkpoints：设置`ckpt_dir`为所需保存目录，可通过以下代码启用

```python
checkpoint = torch.load(checkpoint_path)
style_embedding = checkpoint["embedding_state_dict"]
textual_inversion_model.style_embedding.data = style_embedding
textual_inversion_model.eval()
```
默认训练总步数：total_steps=800
