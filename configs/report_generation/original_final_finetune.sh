export PYTHONPATH=$PYTHONPATH:/path/to/jiawei_workspace/VLM3D_challenge/jiawei_workspace/report_generate_docker/MedM-VL

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29501  /path/to/jiawei_workspace/VLM3D_challenge/jiawei_workspace/report_generate_docker/MedM-VL/lvlm/train.py \
    --deepspeed /path/to/jiawei_workspace/VLM3D_challenge/jiawei_workspace/report_generate_docker/scripts/zero3.json \
    --data_path /path/to/jiawei_workspace/VLM3D_challenge/jiawei_workspace/report_generate_docker/datasets/final_data/final_finetune.json \
    --conv_version qwen3 \
    --image3d_path /path/to/jiawei_database/ctrate_raw \
    --training_recipe common \
    --tune_type_llm full \
    --tune_type_encoder_image full \
    --tune_type_connector_image full \
    --bf16 True \
    --gradient_checkpointing True \
    --output_dir /path/to/jiawei_workspace/VLM3D_challenge/jiawei_workspace/report_generate_docker/exps/final-MedM-VL-siglip-384-qwen3-8b-finetune \
    --resume_from_checkpoint /path/to/jiawei_workspace/VLM3D_challenge/jiawei_workspace/report_generate_docker/exps/final-MedM-VL-siglip-384-qwen3-8b-pretrain \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --eval_strategy no \
    --save_strategy no \
    --report_to tensorboard \
    --logging_steps 1
