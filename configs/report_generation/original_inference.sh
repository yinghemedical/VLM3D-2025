export PYTHONPATH=$PYTHONPATH:/path/to/jiawei_workspace/VLM3D_challenge/jiawei_workspace/report_generate_docker/MedM-VL

MODEL_PATH="/path/to/jiawei_workspace/VLM3D_challenge/jiawei_workspace/report_generate_docker/exps/siglip384-finetune"

CUDA_VISIBLE_DEVICES=1 python /path/to/jiawei_workspace/VLM3D_challenge/jiawei_workspace/report_generate_docker/MedM-VL/lvlm/inference.py \
    --model_dtype bfloat16 \
    --data_path /path/to/jiawei_workspace/feature-fuse/data/raw_rg/infer_rg.json\
    --conv_version qwen3 \
    --image3d_path /path/to/shiym/datasets/CT-RATE/preprocessed_npy/valid/ \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval/ \
    --max_new_tokens 512 \
    --num_beams 1 \
    --temperature 0.1 \
    --batch_size 8
