import torch


for i in range(4):
    a = torch.ones(150, 1024,1920) * i
    data = {"info": {"reset": 1,
                     "actions": 2,
                     "proprioceptive": 4,
                     "exteroceptive": 1913},
            "data": a}
    torch.save(data, "data/"+str(i) + ".pt")
