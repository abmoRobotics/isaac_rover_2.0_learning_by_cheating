import os
import torch
from tqdm import tqdm


def sort_data(path):
    path = path
    files = os.listdir(path)
    if not "data.pt" in files:
        data_path = os.path.join(path, files[0])
        data = torch.load(data_path)
        for i in (pbar := tqdm(range(1, len(files)))):
            pbar.set_description("Processing data")
            data_path = os.path.join(path, files[i])
            data_temp = torch.load(data_path)
            data = torch.cat((data, data_temp), dim=0)
        torch.save(data, path + "data.pt")