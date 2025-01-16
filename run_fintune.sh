python train_lora.py \
    --pretrained_model_name_or_path briaai/BRIA-4B-Adapt \
    --dataset_name Negev900/Modern_Blurred_SeaView \
    --output_dir example_output_lora/ \
    --max_train_steps 1500 \
    --rank 128 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --caption_column prompt \
    --image_column image

