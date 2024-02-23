import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weight_path = "weight/resnet18.ckpt"

weight_as_bytes = open(weight_path, "rb").read()

with open('weight.py', 'w') as f:
    f.write(f'weight_path = "{repr(weight_as_bytes)}"\n')
