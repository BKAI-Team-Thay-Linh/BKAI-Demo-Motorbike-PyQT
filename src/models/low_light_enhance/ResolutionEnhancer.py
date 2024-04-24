# @title Create model
import io

# @title Upload and upscale images or .tar archives
import os
import tarfile

# from google.colab import files
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from RealESRGAN import RealESRGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model_scale = "2"  # @param ["2", "4", "8"] {allow-input: false}

model = RealESRGAN(device, scale=int(model_scale))
model.load_weights(f"weights/resolution/RealESRGAN_x{model_scale}.pth")

IMAGE_FORMATS = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")


def image_to_tar_format(img, image_name):
    buff = BytesIO()
    if ".png" in image_name.lower():
        img = img.convert("RGBA")
        img.save(buff, format="PNG")
    else:
        img.save(buff, format="JPEG")
    buff.seek(0)
    fp = io.BufferedReader(buff)
    img_tar_info = tarfile.TarInfo(name=image_name)
    img_tar_info.size = len(buff.getvalue())
    return img_tar_info, fp


def process_tar(path_to_tar):
    processing_tar = tarfile.open(path_to_tar, mode="r")
    result_tar_path = os.path.join("results/", os.path.basename(path_to_tar))
    save_tar = tarfile.open(result_tar_path, "w")

    for c, member in enumerate(processing_tar):
        print(f"{c}, processing {member.name}")

        if not member.name.endswith(IMAGE_FORMATS):
            continue

        try:
            img_bytes = BytesIO(processing_tar.extractfile(member.name).read())
            img_lr = Image.open(img_bytes, mode="r").convert("RGB")
        except Exception:
            print(f"Unable to open file {member.name}, skipping")
            continue

        img_sr = model.predict(np.array(img_lr))
        # adding to save_tar
        img_tar_info, fp = image_to_tar_format(img_sr, member.name)
        save_tar.addfile(img_tar_info, fp)

    processing_tar.close()
    save_tar.close()
    print(f"Finished! Archive saved to {result_tar_path}")


def process_input(filename):
    if tarfile.is_tarfile(filename):
        process_tar(filename)
    else:
        result_image_path = os.path.join("results/", os.path.basename(filename))
        image = Image.open(filename).convert("RGB")
        sr_image = model.predict(np.array(image))
        sr_image.save(result_image_path)
        print(f"Finished! Image saved to {result_image_path}")
