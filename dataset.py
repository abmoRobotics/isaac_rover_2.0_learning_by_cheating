from torch.utils.data import Dataset
from utils import sort_data
import torch

# TODO add noises to dataset
class TeacherDataset(Dataset):
    def __init__(self, data_dir):
        sort_data(data_dir)
        self.data = torch.load(data_dir + "data.pt")

    def __len__(self):
        return self.data["data"].shape[1]
    
    def __getitem__(self, index):
        info = self.get_info()

        gt = self.data["data"][:,index]
        data = gt
        # TODO add noise



        re, ac, ex = info["reset"], info["actions"], info["exteroceptive"]
        gt_ac = gt[:,re:re+ac]
        gt_ex = gt[:,-ex:]
        return data, gt_ac, gt_ex
    
    def get_info(self):
        return self.data["info"]


if __name__ == "__main__":
    a = TeacherDataset("data/")