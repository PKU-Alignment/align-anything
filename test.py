import torch
print(torch.__version__)
print(torch.cuda.is_available())    # 应输出 True
print(torch.version.cuda)           # 应输出 12.4

import xformers
print(xformers.__version__)