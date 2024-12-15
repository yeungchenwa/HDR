import os
import argparse
from PIL import Image

import torch
from torchvision.transforms import functional as F

from transformers import set_seed
from diffusers import UNet2DModel

from src.build_HDR import build_pipeline
from src.utils import save_image, save_args_to_yaml


def main(args):
    device = args.device

    if args.seed is not None:
        set_seed(args.seed)
    # build unet
    unet = UNet2DModel.from_pretrained(pretrained_model_name_or_path=args.ckpt_path)
    unet = unet.to(device)
    # build pipeline
    pipeline = build_pipeline(
        args=args,
        unet=unet,)
    
    generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)

    assert args.image_path and args.mask_image_path and args.content_image_path, \
        "image_path, mask_image_path and content_image_path should contain."
    image = Image.open(args.image_path).convert('RGB')
    mask_image = Image.open(args.mask_image_path).convert('L')
    content_image = Image.open(args.content_image_path).convert('L')
    image_tensor = F.normalize(F.to_tensor(image), [0.5], [0.5]).unsqueeze(0)
    mask_image_tensor = F.normalize(F.to_tensor(mask_image), [0.5], [0.5]).unsqueeze(0)
    content_image_tensor = F.normalize(F.to_tensor(content_image), [0.5], [0.5]).unsqueeze(0)

    image_tensor = image_tensor.to(device)
    mask_image_tensor = mask_image_tensor.to(device)
    content_image_tensor = content_image_tensor.to(device)

    image = pipeline(
        degraded_image=image_tensor,
        char_mask_image=mask_image_tensor,
        content_image=content_image_tensor,
        image_channel=args.image_channel,
        classifier_free=args.classifier_free,
        content_mask_guidance_scale=args.content_mask_guidance_scale,
        degraded_guidance_scale=args.degraded_guidance_scale,
        generator=generator,
        batch_size=1,
        num_inference_steps=args.num_inference_steps,
        output_type="pil",
    ).images[0]

    # save_result
    image_name = args.image_path.split('/')[-1].split('.')[0]
    image.save(f"{args.save_dir}/{image_name}_repaired_img.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference script for HDR.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--vis_all", action="store_true", \
                        help="Whether to visualize the ori image, mask, content and result on one image.")
    # model
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--image_channel", type=int, default=3)
    # pipeline setting
    parser.add_argument("--pipeline", type=str, default="DPM-Solver++",
                        choices=['DDPM', 'DPM-Solver', 'DPM-Solver++', 'DDIM'])
    parser.add_argument("--classifier_free", action="store_true", \
                        help="Whether to use classifier-free guidance sampling.")
    parser.add_argument(
        "--content_mask_guidance_scale", type=float, default=7.5, help="The guidance scale for contnet and mask image.")
    parser.add_argument(
        "--degraded_guidance_scale", type=float, default=7.5, help="The guidance scale for degraded image.")
    parser.add_argument(
        "--solver_order", type=int, default=2, help="If use DPM-Solver, set this parameter.")
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument("--prediction_type", type=str, default="sample")
    
    # If single image inference, should make sure the image size is 512
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--mask_image_path", type=str, default=None)
    parser.add_argument("--content_image_path", type=str, default=None)
    
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_args_to_yaml(args=args, output_file=f"{args.save_dir}/config.yaml")
    main(args=args)
