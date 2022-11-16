
from dataset import TeacherDataset
from torch.utils.data import DataLoader
from model import Belief_Encoder
Teacher = TeacherDataset("data/")
print(Teacher.__getitem__(0).shape)
trainloader = DataLoader(Teacher,batch_size=4, num_workers=1,pin_memory=1,shuffle=False)
#print(next(iter(trainloader)))
print(iter(trainloader).next().shape)
a = iter(trainloader).next()

#print(a[1])

B = Belief_Encoder()
h = B.init_hidden(4)
print(h.shape)