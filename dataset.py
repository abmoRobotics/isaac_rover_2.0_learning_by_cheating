from torch.utils.data import Dataset
import random
from utils import sort_data
import torch
from heightmap_distribution import Heightmap

# TODO add noises to dataset
class TeacherDataset(Dataset):
    def __init__(self, data_dir):

        self.debug = True

        # Import data and setup variables
        sort_data(data_dir)
        self.data = torch.load(data_dir + "data.pt")
        self.remove_idx = torch.load('remove_idx.pt').to('cpu')
        self.heightmap = Heightmap('cpu')

        self.heightmap_coordinates = self.heightmap.get_coordinates()

        # Tool variables
        self.num_instances = self.data["data"].shape[1]

        # Add initial noise to data
        self.data["data"] = self.add_static_noise(self.data["data"])

        
    def __len__(self):
        return self.data["data"].shape[1]
    
    # Index is the robot instance.
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
        #print(self.remove_idx+7)
        data[:, self.remove_idx+7] = 0
        #print(data.shape)
        #print(data[0,self.remove_idx+7])

        #data[:, self.remove_idx+7] = 0



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
            noise_mode["dev"] = 0.15
            noise_mode["is_add_offset"] = False
            noise_mode["offset"] = 0.0
            noise_mode["is_offset_dev"] = False
            noise_mode["offset_dev"] = False
            noise_mode["is_missing_points"] = True
            noise_mode["missing_points_prob"] = 0.2
        elif r <= 0.9:
            # large offsets
            noise_mode["dev"] = 0.05
            noise_mode["is_add_offset"] = False
            noise_mode["offset"] = 0.0
            noise_mode["is_offset_dev"] = False
            noise_mode["offset_dev"] = 0.02
            noise_mode["is_missing_points"] = True
            noise_mode["missing_points_prob"] = 0.2
        else:
            # large noise magnitude
            noise_mode["dev"] = 0.2
            noise_mode["is_add_offset"] = True 
            noise_mode["offset"] = 0.0
            noise_mode["is_offset_dev"] = True
            noise_mode["offset_dev"] = 0.02
            noise_mode["is_missing_points"] = True
            noise_mode["missing_points_prob"] = 0.2
        return noise_mode

    def create_rand_tensor(self, dev, shape, add_offset=False, offset=0, is_offset_dev=False, offset_dev=0.0):
        # not possible to move height points on xy plane
        
        rand = torch.empty(shape).normal_(mean=0,std=dev)
        # rand = torch.rand(shape)
        # rand = torch.multiply(rand, dev * 2)
        # rand = torch.subtract(rand, dev)
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

    # Add Static noise to data.
    def add_static_noise(self, data):

        if self.debug:
            print(self.num_instances, " instances in dataset")

    # Generate large gaussian holes
        
        # Generate large hole coordinates
        num_large_gaussian = 3
        variance_large_gaussian = 2.0

        large_gaussian_holes = self.random_points(num_large_gaussian)

        distances = self.grid_dist_to_points(large_gaussian_holes[0])

        large_gaussian = self.gaussian_hole(distances, variance_large_gaussian)
        
        data[:,:,7:] = torch.add(data[:,:,7:], -large_gaussian)

    # Generate small gaussian holes

        # Generate small hole coordinates
        num_small_gaussian = 4
        smaller_gaussian_holes = self.random_points_outside(data)


    # Generate moving occlusions

        occlusion_points = self.random_points(3)
        occlusion_distances = self.grid_dist_to_points(occlusion_points[0])
        occlusion = self.occlusion(occlusion_distances, 0.2)

        data[:,:,7:] = data[:,:,7:] * occlusion

        noisy_data = data

        return noisy_data
    
    
    # Generate x random points for each instance, distributed within the heightmap view.
    def random_points(self, num_points):

        # Generate the point coordinates from a random uniform distribution within specified boundaries
        x = torch.FloatTensor(self.num_instances, num_points).uniform_(-2, 2).expand(1, -1,-1)
        y = torch.FloatTensor(self.num_instances, num_points).uniform_(0, 3.5).expand(1, -1,-1)

        # Concatenate the individual coordinates to a combined matrix of coordinates
        rand_points = torch.cat((x,y), 0)

        # Swap the axes to be [num_points, instances, 2(x,y)]
        rand_points = torch.permute(rand_points, (2,1,0))

        return rand_points # Return [num_points, 128, 2(x,y)] tensor

    # Generate a random point for X instances, distributed outside the heightmap view.
    def random_points_outside(self, instances):
        
        num_points = 1

        # Generate the point coordinates from a random uniform distribution
        x = torch.FloatTensor(instances.size()[1], num_points).uniform_(-2, 2)
        y = torch.FloatTensor(instances.size()[1], num_points).uniform_(0, 3.5)

        # Generate bool for placement of border point. True is on the left and right edge. False is on the front and rear edge.
        side = torch.FloatTensor(instances.size()[1], 1).uniform_() > 0.5

        # Set x and y coordinates based on side boolean variable.
        x = torch.where(side, torch.where(x < 0, -2, 2), x)
        y = torch.where(side, y, torch.where(y < 1.5, -1, 4))

        # Concatenate the individual coordinates to a combined matrix of coordinates
        rand_points = torch.cat((x,y), 1)
        
        return rand_points # Return [num_instances, 2(x,y)] tensor

    # Calculate the distance from a single point in each instance heightmap-view(probably 128 of them) to each heightmap point.
    def grid_dist_to_points(self, points):

        a = points # The origin points
        b = self.heightmap_coordinates[:,:2] # The points to measure distance to the "origin points"

        distances = torch.cdist(a.float(), b.float(), p=2.0)

        return distances # Return [instances, 1746] tensor

    # Apply the function of a gaussian hole to the grid_dist points.
    def gaussian_hole(self, dists, variance):

        # Gaussian function
        Gaussian = 1/(variance*torch.sqrt(torch.tensor([3.141592]))) * torch.exp(-1/2 * (dists*dists)/(variance*variance))

        return Gaussian # Return [128, 1746] tensor

    # Apply the occlusion function to the grid_dist points. Dist sets the distance threshold for occlusion/not occlusion.
    def occlusion(self, dist, inflation):

        # Distance funcion. True when not occluded, False when occluded.
        occluded = torch.where(dist > inflation, torch.ones_like(dist), torch.zeros_like(dist))

        return occluded # Return [128, 1746] tensor


if __name__ == "__main__":
    a = TeacherDataset("data/")