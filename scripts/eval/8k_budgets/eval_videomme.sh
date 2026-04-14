# ==========================================
# 1. Data & Model Configuration
# ==========================================
video_fps=2
min_frames_num=8
max_frames_num=1024
model_max_length=32768
frame_windows=8
frame_stride=8
conv_version="qwen"

# ATA Configuration
visual_token_budget=8192
disable_dynamic_compress="False"

# ==========================================
# 2. Experiment Configuration
# ==========================================
model_name="Tempo-6B-8K"
model_path="./checkpoints/Tempo-6B"

output_path="./results/${model_name}/"
mkdir -p "$output_path"

exp="${model_name}_frame${min_frames_num}_${max_frames_num}_fps${video_fps}_win${frame_windows}_budget${visual_token_budget}"

# ==========================================
# 3. Task List
# ==========================================
# tasks=(lvbench mlvu_dev videomme longvideobench_val_v)
tasks=(videomme)

# ==========================================
# 4. Run
# ==========================================
NUM_GPUS=4

for task in "${tasks[@]}"; do
    echo "================================================================"
    echo "🚀 Running task: $task"
    echo "================================================================"

    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=$NUM_GPUS \
        -m lmms_eval \
        --model tempo \
        --model_args "pretrained=$model_path,video_fps=$video_fps,min_frames_num=$min_frames_num,max_frames_num=$max_frames_num,model_max_length=$model_max_length,frame_windows=$frame_windows,frame_stride=$frame_stride,conv_version=$conv_version,visual_token_budget=$visual_token_budget,disable_dynamic_compress=$disable_dynamic_compress" \
        --tasks "$task" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "$exp" \
        --output_path "$output_path"
    
    echo "✅ Task $task completed. Logs saved to $output_path"
done