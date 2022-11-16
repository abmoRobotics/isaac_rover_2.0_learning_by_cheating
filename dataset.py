from torch.utils.data import Dataset
from utils import sort_data
import torch

# TODO add noises to dataset
class TeacherDataset(Dataset):
    def __init__(self, data_dir):
        sort_data(data_dir)
        self.data = torch.load(data_dir + "data.pt")

    def __len__(self):
        return self.data.shape[1]
    
    def __getitem__(self, index):
        gt = self.data[:,index]
        data = gt
        # TODO add noise
        
        return data, gt
    


if __name__ == "__main__":
    a = TeacherDataset("data/")