from src.hdr_pipeline import HDRPipeline
from src.unet import UNet2DModel
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler)

def build_model(args):
    model = UNet2DModel(
        skip_out=args.skip_out,
        sample_size=args.resolution,
        in_channels=8, # noisy_images, degraded_images, degraded_total_content_images, degraded_total_masks
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    return model

def build_pipeline(args, unet):
    if args.pipeline == 'DDPM':
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,)
    elif args.pipeline == 'DPM-Solver':
        noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            solver_order=args.solver_order,
            prediction_type=args.prediction_type,
            algorithm_type="dpmsolver",)
    elif args.pipeline == 'DPM-Solver++':
        noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            solver_order=args.solver_order,
            prediction_type=args.prediction_type,
            algorithm_type="dpmsolver++",)
    elif args.pipeline == 'DDIM':
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,)
    else:
        raise f"There is no suitable pipeline {args.pipeline}"
    
    return HDRPipeline(
        unet=unet,
        scheduler=noise_scheduler,)