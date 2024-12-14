from typing import List, Optional, Tuple, Union

import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline, ImagePipelineOutput


class HDRPipeline(DiffusionPipeline):
    r"""
    Pipeline for Historical Document Restoration.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        degraded_image: torch.Tensor,
        char_mask_image: torch.Tensor,
        content_image: torch.Tensor,
        image_channel: int,
        batch_size: int = 1,
        classifier_free: bool = False,
        content_mask_guidance_scale: float = None,
        degraded_guidance_scale: float = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            degraded_image: (torch.Tensor),
            char_mask_image: (torch.Tensor),
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                image_channel,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, image_channel, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        if classifier_free:
            # generate the uncondition image
            uncond_char_mask_image = -torch.ones_like(char_mask_image).to(degraded_image.device)
            uncond_content_image = torch.ones_like(content_image).to(degraded_image.device)
            uncond_degraded_image = torch.ones_like(degraded_image).to(degraded_image.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. processing the input
            if not classifier_free:
                # (1) without classifier-free guidance
                unet_input = torch.cat([image, degraded_image, content_image, char_mask_image], dim=1)
            else:
                # (2) with classifier-free guidance
                cond_degraded_cond_cm = torch.cat([image, degraded_image, content_image, char_mask_image], dim=1)
                cond_degraded_uncond_cm = torch.cat([image, degraded_image, uncond_content_image, uncond_char_mask_image], dim=1)
                uncond_degraded_uncond_cm = torch.cat([image, uncond_degraded_image, uncond_content_image, uncond_char_mask_image], dim=1)
                unet_input = torch.cat([cond_degraded_cond_cm, cond_degraded_uncond_cm, uncond_degraded_uncond_cm], dim=0)
            
            # 2. Input to the UNet
            model_output = self.unet(unet_input, t).sample

            # 3. Processing the output if classifier-free guidance
            if classifier_free:
                cond_degraded_cond_cm_out, cond_degraded_uncond_cm_out, uncond_degraded_uncond_cm_out = torch.chunk(model_output, chunks=3, dim=0)
                model_output = uncond_degraded_uncond_cm_out + \
                                degraded_guidance_scale * (cond_degraded_uncond_cm_out - uncond_degraded_uncond_cm_out) + \
                                content_mask_guidance_scale * (cond_degraded_cond_cm_out - cond_degraded_uncond_cm_out)

            # 4. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


