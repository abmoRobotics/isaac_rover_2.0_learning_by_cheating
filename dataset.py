from torch.utils.data import Dataset
import random
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
        max_delay = 0
        info = self.get_info()
        gt = self.data["data"][:, index]
        data = self.add_noise(gt)
        # shift actions to simulate random delay for whole rover
        delay = random.randint(0, max_delay)
        re, ac, ex = info["reset"], info["actions"], info["sparse"] + info["dense"]
        actions_delayed = torch.roll(data[:, re:re + ac], -delay, 0)
        actions_delayed = actions_delayed[:-(max_delay+1), :]
        data = data[:-(max_delay+1), :]
        gt = gt[:-(max_delay+1), :]
        data[:, re:re + ac] = actions_delayed
        gt_ac = gt[:, re:re + ac]
        gt_ex = gt[:, -ex:]
        #data = gt
        return data, gt_ac, gt_ex

    def add_noise(self, gt):
        # starting with index 3: 0: reset bit 1: action1 2: action2
        noisy_data = gt.clone()
        # dist2goal
        noise = self.create_rand_tensor(0.0, noisy_data[:, 3].shape)
        noisy_data[:, 3] = torch.add(noisy_data[:, 3], noise)
        # heading2goal
        noise = self.create_rand_tensor(0.0, noisy_data[:, 4].shape)
        noisy_data[:, 4] = torch.add(noisy_data[:, 4], noise)
        # linear velocity
        noise = self.create_rand_tensor(0.0, noisy_data[:, 5].shape)
        noisy_data[:, 5] = torch.add(noisy_data[:, 5], noise)
        # angular velocity
        noise = self.create_rand_tensor(0.0, noisy_data[:, 6].shape)
        noisy_data[:, 6] = torch.add(noisy_data[:, 6], noise)
        # heightmap
        noise_mode = self.get_noise_mode()
        noise = self.create_rand_tensor(noise_mode["dev"],
                                        noisy_data[:, 7:].shape,
                                        add_offset=noise_mode["is_add_offset"],
                                        offset=noise_mode["offset"],
                                        is_offset_dev=noise_mode["is_offset_dev"],
                                        offset_dev=noise_mode["offset_dev"])
        #print("NOISE: " + str(noise))
        noisy_data[:, 7:] = torch.add(noisy_data[:, 7:], noise)
        if noise_mode["is_missing_points"]:
            noisy_data[:, 7:] = self.simulate_missing_height_points(noisy_data[:, 7:],
                                                                    noise_mode["missing_points_prob"])
        return noisy_data

    def get_noise_mode(self):
        noise_mode = {}
        r = random.random()
        if r <= 0.6:
            # normal noise
            noise_mode["dev"] = 0.2
            noise_mode["is_add_offset"] = False
            noise_mode["offset"] = 0.0
            noise_mode["is_offset_dev"] = False
            noise_mode["offset_dev"] = False
            noise_mode["is_missing_points"] = True
            noise_mode["missing_points_prob"] = 0.1
        elif r <= 0.9:
            # large offsets
            noise_mode["dev"] = 0.2
            noise_mode["is_add_offset"] = False #True
            noise_mode["offset"] = 5.0
            noise_mode["is_offset_dev"] = False #True
            noise_mode["offset_dev"] = 1.0
            noise_mode["is_missing_points"] = True
            noise_mode["missing_points_prob"] = 0.1
        else:
            # large noise magnitude
            noise_mode["dev"] = 0.3
            noise_mode["is_add_offset"] = False 
            noise_mode["offset"] = 0.0
            noise_mode["is_offset_dev"] = False
            noise_mode["offset_dev"] = False
            noise_mode["is_missing_points"] = True
            noise_mode["missing_points_prob"] = 0.1
        return noise_mode

    def create_rand_tensor(self, dev, shape, add_offset=False, offset=0, is_offset_dev=False, offset_dev=0.0):
        # not possible to move height points on xy plane
        
        rand = torch.empty(shape).normal_(mean=0,std=dev)
        #rand = torch.rand(shape)
        #rand = torch.multiply(rand, dev * 2)
        #rand = torch.subtract(rand, dev)
        #print(rand.shape, rand.mean())
        if add_offset:
            if is_offset_dev:
                offset = torch.rand(shape)
                offset = torch.multiply(offset, offset_dev * 2)
                offset = torch.subtract(offset, dev)
                torch.add(rand, offset)
            else:
                torch.add(rand, offset)
        return rand

    def simulate_missing_height_points(self, heights, missing_point_probability):
        rand = torch.rand(heights.shape)
        new_heights = torch.where(rand <= missing_point_probability, 0, heights)
        return new_heights
    
    def get_info(self):
        # print("KEYS ", self.data["info"].keys())
        return self.data["info"]


if __name__ == "__main__":
    a = TeacherDataset("data/")