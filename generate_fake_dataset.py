import torch


for i in range(4):
    a = torch.ones(150, 1024,1920) * i
    torch.save(a,"data/"+str(i) + ".pt")

