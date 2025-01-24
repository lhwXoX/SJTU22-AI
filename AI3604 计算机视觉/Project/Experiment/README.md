<h1 align='center'>Computer Vision Final Project</h1>
<h4 align='center'>Medical Segmentation Based on SAM and Implementation of Variant Problems</h4>

---
**Group 8**
**Members: Zhi Han, Nan Jiang, Hanwen Liu, Jingyao Wu**


## a): fine-tune on FLARE22 dataset
### Step 0: setup environment
If using conda environment:
```bash
conda env create -f environment.yml
```
If directly using pip
```bash
pip install -r requirements.txt
```
### Step 1: dataset preperation:
First, download Flare22 dataset at [Flare22](https://flare22.grand-challenge.org/Dataset/). Then, preprocess the 3D dataset and save images/masks as 2D slices.

There is no strict format for dataset folder; you just need to identify your main dataset folder, for example:
```python
args.img_folder = './datasets/'
args.mask_folder = './datasets/'
```
Next, prepare the image/mask list file train/val/test.csv under **args.img_folder/dataset_name/** in the following format: ``img_slice_path mask_slice_path``, such as:
```
E:/Homework/CV_Project/finetune-SAM/datasets/Flare/images/FLARETs_0020_126.png,E:/Homework/CV_Project/finetune-SAM/datasets/Flare/masks/FLARETs_0020_126.png
E:/Homework/CV_Project/finetune-SAM/datasets/Flare/images/FLARETs_0010_560.png,E:/Homework/CV_Project/finetune-SAM/datasets/Flare/masks/FLARETs_0010_560.png
E:/Homework/CV_Project/finetune-SAM/datasets/Flare/images/FLARETs_0015_342.png,E:/Homework/CV_Project/finetune-SAM/datasets/Flare/masks/FLARETs_0015_342.png
```
We provide a python code ``nii2png_prep.py`` to automatically generate the corresponding 2D slices and CSV file. To run the code, you need to first modify line 11-16, line 95-97 to your personal path.

## Step 2:
Configure your network architectures and other hyperparameters.
### (1) Choose pretrained SAM model.
In our experiment, we choose ``vit_b`` as our base model.
```python
args.arch = 'vit_b' #'vit_h', 'vit_b', 'vit_t'
args.sam_ckpt = "sam_vit_b_01ce64.pth" #if choose vit_b, otherwise, replace the name to your model
```
SAM's checkpoints can be downloaded from [SAM](https://github.com/facebookresearch/segment-anything).

### (2) Choose fine-tuning Methods.

#### (i) fine-tuning using LoRA blocks
In our experiment, we add LoRA layers on the image encoder and mask decoder both:
```python
args.if_update_encoder = True
args.if_encoder_lora_layer = True
args.encoder_lora_layer = [] # add LoRA layer to each transformer block
args.if_decoder_lora_layer = True
```
#### (ii) fine-tuning using Adapter blocks
We add Adapter blocks on the image encoder and mask decoder both:
```python
args.if_mask_decoder_adapter = True
args.if_update_encoder = True
args.if_encoder_adapter = True
args.encoder_adapter_depths = range(0,12) # All layers
args.if_mask_decoder_adapter=True
```
### Other configuration
Flare22 dataset consists of 13 classes and 1 background class. In order to do multi-class segmentation:
```python
args.num_cls = 14 #13 organ class + 1 background class
args.targets = 'multi_all'
args.dataset_name = 'Flare' # replace the name as your dataset's name
```
We train each of our model for 50 epochs:
```python
args.epochs = 50
```
See more configuration in ``cfg.py``.
### Bash file for training and evaluation
We provide two bash file ``train_mulclass.sh`` and ``val_mul.sh`` .
```bash
#!/bin/bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES="0"

# Define variables
arch="vit_b"  # Change this value as needed
finetune_type="lora"
dataset_name="GID-MTL"  # Assuming you set this if it's dynamic
targets='multi_all' # make it as binary segmentation 'multi_all' for multi cls segmentation
# Construct train and validation image list paths
img_folder="./datasets"  # Assuming this is the folder where images are stored
train_img_list="${img_folder}/${dataset_name}/train.csv"
val_img_list="${img_folder}/${dataset_name}/val.csv"

# Construct the checkpoint directory argument
dir_checkpoint="2D-SAM_${arch}_encoder_decoder_${finetune_type}_${dataset_name}_noprompt"

# Run the Python script
python train_finetune_noprompt.py \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -if_update_encoder True \
    -if_encoder_lora_layer True \
    -if_decoder_lora_layer True \
    -epochs 50\
    -img_folder "$img_folder" \
    -mask_folder "$img_folder" \
    -sam_ckpt "./sam_vit_b_01ec64.pth" \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint" \
    -train_img_list "$train_img_list" \
    -val_img_list "$val_img_list" \
    -num_cls 14\
    -targets "$targets"\
```
```bash
#!/bin/bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES="0"

# Define variables
arch="vit_b"  # Change this value as needed
finetune_type="lora"
dataset_name="Flare"  # Assuming you set this if it's dynamic
targets='multi_all'
# Construct the checkpoint directory argument
dir_checkpoint="SAM_Flare_Lora=4"

# Run the Python script
python val_mul_noprompt.py \
    -if_warmup True \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -if_update_encoder True \
    -if_encoder_lora_layer True \
    -if_decoder_lora_layer True \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint"\
    -targets "$targets"
```
The ``val_mul.sh`` calculates DSC on each classes and visualizes 20 predicted masks. You can customize the configuration in the bash file. Remember to modify ``dir_checkpoint`` to the folder of your fine-tuned model and your model's name should be ``checkpoint_best.pth``. To run the training and evaluation, just use the command:
```bash
bash train_mulclass.sh
bash val_mul.sh
```
---
## b): fine-tune on downstream scenario
### (1) fine-tuning on GID5
Download GID dataset at [GID-MTL5/15](https://github.com/cheer00/GID-MTL). We use GID-MTL5 datasets in our experiment which only consists of 6 classes(5 landscapes and 1 background class).
To modify SAM to train on GID-MTL5:
```python
args.num_cls = 6
```
### (2) fine-tuning on SIRST
Download SIRST dataset at [SIRST5k](https://pan.baidu.com/share/init?surl=EG-loK86aWJL7M6bPQjivA&pwd=1234). Different from Flare and GID, SIRST only consists of 2 classes(1 small object class and 1 background class).
To modify SAM to train on SIRST:
```python
args.num_cls = 2
args.targets = 'combine_all'
```
---
## c): generation and visualization of SAM/SAM2
### (1) prepare model and data
See more details in the official github page [SAM](https://github.com/facebookresearch/segment-anything) and [SAM2](https://github.com/facebookresearch/sam2) for preparing model. Make sure the input images are in an image folder.
### (2) generate and visualize the predicted mask
We provide two python codes to automatically generate and visualize the predicted masks given by SAM/SAM2. To run the python code, you just need to run the following command:
```
python SAM.py
python SAM2.py
```
You may modify the path to your image folder and the path to the output folder. The three paths in the code are:
```python
images_folder = 'images_folder' #原图文件夹地址
output_folder_masks = 'output_folder_by_SAM/SAM2' #SAM/SAM2结果文件夹地址
output_folder = 'output_folder' #可视化输出结果文件夹
```

---
## d): other file
- Our fine-tuned SAM(LoRA, Adapter) can be downloaded at [jbox](https://jbox.sjtu.edu.cn/l/Y1gryB).
- result.xlsx: the test dsc of each model.
- log: training log, it can be downloaded at [jbox](https://jbox.sjtu.edu.cn/l/tH1YwA).
- Visualization: our visualization result of each model, it can be downloaded at [jbox](https://jbox.sjtu.edu.cn/l/Z1jOSt).
- PPT.
- Report.pdf: our final report.
---
## Acknowlegdement
We build our codes based on:
1. [finetune-SAM](https://github.com/mazurowski-lab/finetune-SAM)
2. [Medical SAM Adapter](https://github.com/KidsWithTokens/Medical-SAM-Adapter)
3. [LoRA for SAM](https://github.com/JamesQFreeman/Sam_LoRA)