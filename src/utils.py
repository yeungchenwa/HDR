import yaml
import numpy as np
from PIL import Image
from typing import Optional

from huggingface_hub import HfFolder, whoami

import torch


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    arr = arr.to(timesteps.device)
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def get_transform_images(augmentations):
    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}
    return transform_images

def get_model_para_size(model):
    total = sum([param.nelement() for param in model.parameters()])
    print(f"Number of parameter: {total/1e6}M")

def save_args_to_yaml(args, output_file):
    # Convert args namespace to a dictionary
    args_dict = vars(args)

    # Write the dictionary to a YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)

def tensor2numpy_batch(tensor_image, channel=3):
    tensor_image = (tensor_image / 2 + 0.5).clamp(0, 1)
    out_image = tensor_image.cpu().permute(0, 2, 3, 1).numpy()
    out_image = (out_image * 255).round().astype("uint8")
    if channel == 1:
        out_image = np.repeat(out_image, 3, axis=3)
    return out_image

def numpy2pil(numpy_images_list):
    return [Image.fromarray(image) for image in numpy_images_list]

def save_image(save_dir, image_names, degraded_images, char_mask_images, \
               content_images, output_images, gt_images=None):
    # tensor to pil
    degraded_images = tensor2numpy_batch(degraded_images, channel=3)
    char_mask_images = tensor2numpy_batch(char_mask_images, channel=1)
    content_images = tensor2numpy_batch(content_images, channel=1)

    if gt_images is not None:
        gt_images = tensor2numpy_batch(gt_images, channel=3)
        for i_name, d_image, m_image, c_image, o_image, g_image in \
            zip(image_names, degraded_images, char_mask_images, \
               content_images, output_images, gt_images):
            h, w = d_image.shape[0], d_image.shape[1]
            new_image = Image.new('RGB', (w * 5, h), (255, 255, 255))
            d_image, m_image, c_image, g_image = \
                numpy2pil([d_image, m_image, c_image, g_image])
            new_image.paste(m_image, (0, 0, w, h))
            new_image.paste(c_image, (w, 0, w * 2, h))
            new_image.paste(d_image, (w * 2, 0, w * 3, h))
            new_image.paste(o_image, (w * 3, 0, w * 4, h))
            new_image.paste(g_image, (w * 4, 0, w * 5, h))
            # save image
            new_image.save(f"{save_dir}/{i_name}")
    else:
        for i_name, d_image, m_image, c_image, o_image in \
            zip(image_names, degraded_images, char_mask_images, \
               content_images, output_images):
            h, w = d_image.shape[0], d_image.shape[1]
            new_image = Image.new('RGB', (w * 4, h), (255, 255, 255))
            d_image, m_image, c_image = \
                numpy2pil([d_image, m_image, c_image])
            new_image.paste(m_image, (0, 0, w, h))
            new_image.paste(c_image, (w, 0, w * 2, h))
            new_image.paste(d_image, (w * 2, 0, w * 3, h))
            new_image.paste(o_image, (w * 3, 0, w * 4, h))
            # save image
            new_image.save(f"{save_dir}/{i_name}")
