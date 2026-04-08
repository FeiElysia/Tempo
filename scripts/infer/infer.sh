#!/bin/bash
# ==================================================================
# Tempo-6B: Single Video Inference
# ==================================================================

MODEL_PATH="./checkpoints/Tempo-6B"

DEFAULT_QUERY="Task: Please analyze the provided video and answer the following 7 questions precisely.
Q1: How many performers are visible on the stage?
Q2: Describe the architectural elements in the background. What historical civilization do they remind you of?
Q3: What is happening in the night sky above the performers, and what does this suggest about the event?
Q4: List the hair colors of the performers in order from left to right.
Q5: Identify the specific musical instrument being played by the performer located on the far left of the stage.
Q6: What is the specific time interval (in seconds, e.g., XX-XXs) during which this fireworks performance scene occurs in the video?
Q7: Look at the audience in the foreground. How does their silhouette-like depiction affect the viewer's perspective of the stage?"

VIDEO_PATH=${1:-"./examples/hsr_helloworld.mp4"}
QUERY=${2:-"$DEFAULT_QUERY"}

echo "🚀 Running Tempo-6B Inference..."
echo "📁 Video: $VIDEO_PATH"
echo "❓ Query: $QUERY"
echo "------------------------------------------------------------------"

python infer.py \
    --model_path "$MODEL_PATH" \
    --video_path "$VIDEO_PATH" \
    --query "$QUERY"

echo "✅ Inference Completed!"