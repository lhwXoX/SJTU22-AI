#from segment_anything import SamPredictor, sam_model_registry
from models.sam import SamPredictor, sam_model_registry
from models.sam.utils.transforms import ResizeLongestSide
from skimage.measure import label
from models.sam_LoRa import LoRA_Sam
#Scientific computing 
import numpy as np
import os
#Pytorch packages
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import datasets
#Visulization
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
#Others
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy
from utils.dataset import Public_dataset
import torch.nn.functional as F
from torch.nn.functional import one_hot
from pathlib import Path
from tqdm import tqdm
from utils.losses import DiceLoss
from utils.dsc import dice_coeff
import cv2
import monai
from utils.utils import vis_image
import cfg
from argparse import Namespace
import json
import random
import tensorboard
import shutil

def visualization(image_path, mask_path, save_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    
    result = image.copy()
    non_zero_indices = mask > 0  # 物体区域
    #non_zero_indices = np.stack([non_zero_indices] * 3, axis=-1)
    #print(non_zero_indices.shape)
    #print(result.shape)
    #print(colored_mask.shape)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            for k in range(result.shape[2]):
                if non_zero_indices[i, j, k] == 0:
                    result[i, j, k] = image[i, j, k]
                else:
                    result[i, j, k] = image[i, j, k] * 0.7 + colored_mask[i, j, k] * 0.3
    cv2.imwrite(save_path, result)

def main(args,test_img_list):
    # change to 'combine_all' if you want to combine all targets into 1 cls
    test_dataset = Public_dataset(args,args.img_folder, args.mask_folder, test_img_list,phase='val',targets=[args.targets],if_prompt=False)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    if args.finetune_type == 'adapter' or args.finetune_type == 'vanilla':
        sam_fine_tune = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.dir_checkpoint,'checkpoint_best.pth'),num_classes=args.num_cls)
    elif args.finetune_type == 'lora':
        sam = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.sam_ckpt),num_classes=args.num_cls)
        sam_fine_tune = LoRA_Sam(args,sam,r=4).to('cuda').sam
        sam_fine_tune.load_state_dict(torch.load(args.dir_checkpoint + '/checkpoint_best.pth'), strict = False)
        
    sam_fine_tune = sam_fine_tune.to('cuda').eval()
    class_iou = torch.zeros(args.num_cls,dtype=torch.float)
    cls_dsc = torch.zeros(args.num_cls,dtype=torch.float)
    eps = 1e-9
    img_name_list = []
    pred_msk = []
    test_img = []
    test_gt = []

    for i,data in enumerate(tqdm(testloader)):
        imgs = data['image'].to('cuda')
        msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask'])
        #print('mask:', np.unique(msks))
        msks = msks.to('cuda')
        img_name_list.append(data['img_name'][0])

        with torch.no_grad():
            img_emb= sam_fine_tune.image_encoder(imgs)

            sparse_emb, dense_emb = sam_fine_tune.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
            pred_fine, _ = sam_fine_tune.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=sam_fine_tune.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb, 
                            multimask_output=True,
                          )
        #print(pred_fine.shape)
        pred_fine = pred_fine.argmax(dim=1)

        
        pred_msk.append(pred_fine.cpu())
        test_img.append(imgs.cpu())
        test_gt.append(msks.cpu())
        yhat = (pred_fine).cpu().long().flatten()
        y = msks.cpu().flatten()

        for j in range(args.num_cls):
            y_bi = y==j
            yhat_bi = yhat==j
            I = ((y_bi*yhat_bi).sum()).item()
            U = (torch.logical_or(y_bi,yhat_bi).sum()).item()
            class_iou[j] += I/(U+eps)

        for cls in range(args.num_cls):
            mask_pred_cls = ((pred_fine).cpu()==cls).float()
            mask_gt_cls = (msks.cpu()==cls).float()
            cls_dsc[cls] += dice_coeff(mask_pred_cls,mask_gt_cls).item()
        #print(i)

    class_iou /=(i+1)
    cls_dsc /=(i+1)
    num_samples = 20
    random.seed(42)
    indices = random.sample(range(len(pred_msk)), min(num_samples, len(pred_msk)))
    for i in indices:
        #print(pred_msk[0].shape)
        pred = pred_msk[i].squeeze(0).numpy().astype(np.uint8) #use 0 for demonstration!
        #print(pred)
        #print(np.unique(pred))
        save_folder = os.path.join('test_results',args.dir_checkpoint)
        Path(save_folder).mkdir(parents=True,exist_ok = True)
        #np.save(os.path.join(save_folder,'test_masks.npy'),np.concatenate(pred_msk,axis=0))
        #np.save(os.path.join(save_folder,'test_name.npy'),np.concatenate(np.expand_dims(img_name_list,0),axis=0))
        Image.fromarray((pred/pred.max() * 255).astype(np.uint8)).save(os.path.join(save_folder, f'pred mask {i}.png'))

        src_path = img_name_list[i]
        print(src_path)
        mask_path = src_path.replace("images/", "masks/")
        print(mask_path)
        dst_path = os.path.join(save_folder, f'input image {i}.png')
        dst_msk_path = os.path.join(save_folder, f'gt mask {i}.png')
        mask_image = Image.open(mask_path)
        mask_image = np.array(mask_image)
        Image.fromarray((mask_image/mask_image.max() * 255).astype(np.uint8)).save(dst_msk_path)
        shutil.copy(src_path, dst_path)
        #shutil.copy(mask_path, dst_msk_path)
        #Visualization
        visualization(src_path, dst_msk_path, os.path.join(save_folder, f'gt result {i}.png'))
        visualization(src_path, os.path.join(save_folder, f'pred mask {i}.png'), os.path.join(save_folder, f'pred result {i}.png'))
    print(save_folder)
    print(dataset_name)      
    print('class dsc:',cls_dsc)      
    print('class iou:',class_iou)
    
if __name__ == "__main__":
    args = cfg.parse_args()

    if 1: # if you want to load args from taining setting or you want to identify your own setting
        args_path = f"{args.dir_checkpoint}/args.json"

        # Reading the args from the json file
        with open(args_path, 'r') as f:
            args_dict = json.load(f)
        
        # Converting dictionary to Namespace
        args = Namespace(**args_dict)
        
    dataset_name = args.dataset_name
    print('train dataset: {}'.format(dataset_name)) 
    test_img_list =  args.img_folder + f'/{dataset_name}' + '/test.csv'
    main(args,test_img_list)