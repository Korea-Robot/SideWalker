from erfpspnet import Net
import torch

model = Net(22)

model_path = "./model_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(model_path, map_location=device)

# Remove 'module.' if present
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.', '') if k.startswith('module.') else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()



x = torch.randn(4, 3, 480, 640)
x = x.to(device)
y= model(x)
breakpoint()