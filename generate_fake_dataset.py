import torch


for i in range(4):
    a = torch.ones(150, 1024,9) * i *0.25
    for j in range(150):
        b = torch.ones(1024,9)
        a[j] = torch.sin(b*j*j*0.01)
    
    #print(a[:,0])
    data = {"info": {"reset": 1,
                     "actions": 2,
                     "proprioceptive": 4,
                     "exteroceptive": 2},#},1913},
            "data": a}
    torch.save(data, "data/"+str(i) + ".pt")
