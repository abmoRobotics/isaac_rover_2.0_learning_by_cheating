# Isaac rover 2.0 learning by cheating
This repository contains the code needed to train a student agent based on data logged from running the teacher agent in simulation. When starting the training process, the ground truth data of the teacher is modulated with noise of three different modes and supervised learning is used to train the student policy. Because of the structure of the student network, it learns to reconstruct the heightmap, thus filtering out noise and making the policy more robus. 

For more details please refer to our paper.

