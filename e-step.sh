source /gpfs/share/software/anaconda/3-2023.09-0/etc/profile.d/conda.sh
conda activate DPS

python3 sample_condition.py \
    --model_path=checkpoint/last_m_step.pt \
    --model_config=configs_ddpm/model_config.yaml \
    --diffusion_config=configs_ddpm/diffusion_config.yaml \
    --task_config=configs_ddpm/inpainting_128.yaml \
    --gpu=0 \
    --shuffle=0 \
    --use_gt_model=0 \
    --gt_model_name=google/ddpm-cifar10-32 \
    --batch_size=250 \
    --batch_num=1 \
    --start=0 \
    --seed=0 \
    --save_dir=./exp/result

