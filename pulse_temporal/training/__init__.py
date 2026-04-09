# Training module for PULSE temporal awareness
#
# Data generation:
#   python -m pulse_temporal.training.data_generator
#
# Fine-tuning:
#   python -m pulse_temporal.training.temporal_tuner \
#     --model Qwen/Qwen2.5-1.5B-Instruct \
#     --data data/temporal_train.jsonl \
#     --output models/pulse-qwen-1.5b
