python inference.py \
    --nav_model_path checkpoints/navigation_model_epoch_100.pth \
    --world_model_path checkpoints/world_model_epoch_100.pth \
    --dataset_path ../world_data/imitation_dataset \
    --k_steps 20 \
    --sample_idx 5 \
    --output_dir inference_results