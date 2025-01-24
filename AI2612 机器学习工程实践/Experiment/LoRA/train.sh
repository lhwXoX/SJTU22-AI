export MODEL_NAME="./stable-diffusion"

nohup accelerate launch --num_processes 2 --gpu_ids 4,6 --mixed_precision="no"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="./data/shuimo" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=5000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="./output/shuimo" \
  --allow_tf32 \
  --scale_lr \
  --rank=4 \
  > logs/shuimo.log 2>&1 &\

wait

nohup accelerate launch --num_processes 2 --gpu_ids 4,6 --mixed_precision="no"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="./data/comic" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=5000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="./output/comic" \
  --allow_tf32 \
  --scale_lr \
  --rank=4 \
  > logs/comic.log 2>&1 &\

wait

nohup accelerate launch --num_processes 2 --gpu_ids 4,6 --mixed_precision="no"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="./data/graphic_design" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=5000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="./output/graphic_design" \
  --allow_tf32 \
  --scale_lr \
  --rank=4 \
  > logs/graphic_design.log 2>&1 &\

