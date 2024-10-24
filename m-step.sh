source /gpfs/share/software/anaconda/3-2023.09-0/etc/profile.d/conda.sh
conda activate DPS

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -u -m accelerate.commands.launch denoiser_diffusion.py --exp training --dataset_path path_to_dataset --model_path path_to_checkpoint --ema


