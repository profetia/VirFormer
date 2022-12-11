import torch

model = torch.load('../../../VirFormer/fulldata_lr_1e-3_5000_20_3layers.pt')

torch.save(model.state_dict(), './fulldata_lr_1e-3_5000_20_3layers.pth')