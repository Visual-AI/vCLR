import torch 
# torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo

repo = "isl-org/ZoeDepth"
model = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
model = model.cuda(0)

from PIL import Image
DEVICE="cuda:0" 
from zoedepth.utils.misc import pil_to_batched_tensor
from tqdm import tqdm
import numpy as np
import os

dirname = 'datasets/coco/val2017'
names = os.listdir(dirname)

for name in tqdm(names):
    imgname = os.path.join(dirname, name)
    image = Image.open(imgname).convert("RGB")  # load
    X = pil_to_batched_tensor(image).to(DEVICE)
    depth_tensor = model.infer(X)
    depth = depth_tensor.detach().cpu().numpy()[0][0]
    np.save('datasets/val2017_depth/' + name[:-4] + '.npy', depth)