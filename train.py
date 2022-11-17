import torch
from torch import nn
from torch import optim
from model import Student
from tqdm import tqdm
from dataset import TeacherDataset
from torch.utils.data import DataLoader
import datetime
import wandb
import os

class Trainer():
    def __init__(self):
        super(Trainer,self).__init__()
        self.LEARNING_RATE = 1e-4 # original 1e-4
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_EPOCHS = 10
        self.RUN_NAME = "TEST"
        self.BATCH_SIZE = 4
    
        self.wandb_group = "test-group"
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.wandb_name = f"test-run_{time_str}"

    def train_fn(self, train_loader, model, optimizer, loss_fn, scaler):
        loop = tqdm(train_loader)
        #TODO add metrics
        total_loss = 0
        total_be_loss = 0
        total_re_loss = 0

        h = model.belief_encoder.init_hidden(self.BATCH_SIZE).to(self.DEVICE)
        
        for batch_idx, (data, targets_ac, targets_ex) in enumerate(loop):
            data = data.to(device=self.DEVICE)

            #TODO format target to be in the correct format
            targets_ac = targets_ac.float().to(device=self.DEVICE)
            targets_ex = targets_ex.float().to(device=self.DEVICE)
           # print(targets_ac[:,0])
            
            # forward
            with torch.cuda.amp.autocast():
                actions, predictions = model(data,h)
                loss_be = loss_fn["behaviour"](actions, targets_ac)
                loss_re = loss_fn["recontruction"](predictions, targets_ex)
                loss = loss_be + (0.5 * loss_re)
                
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # update tqdm loop
            loop.set_postfix(loss=loss.item())
            
            total_loss += loss.item()
            total_be_loss += loss_be
            total_re_loss += loss_re

        return total_loss, total_be_loss, total_re_loss

    def train(self):
        wandb.init(project='isaac-rover-2.0-learning-by-cheating', sync_tensorboard=True,name=self.wandb_name,group=self.wandb_group, entity="aalborg-university")
        train_ds = TeacherDataset("data/")
        train_loader = DataLoader(train_ds,batch_size=self.BATCH_SIZE,num_workers=4,pin_memory=True, shuffle=False)
        print(len(train_loader))
        model = Student(train_ds.get_info()).to(self.DEVICE)
        loss_fn = {
            "behaviour":  nn.MSELoss(reduction="mean"),
            "recontruction": nn.MSELoss(reduction="mean")
        }
        optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
        scaler = torch.cuda.amp.GradScaler()


        best = float('inf')

        for epoch in range(0,self.NUM_EPOCHS):
            
            loss, loss_be, loss_re = self.train_fn(train_loader, model, optimizer, loss_fn, scaler)
            
             # Reset hidden units after each epoch

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            # # TODO calculate metrics
            # metrics = 0

            if loss < best:
                print("Best model found => saving")
                best = loss
                self.save_checkpoint(checkpoint,self.wandb_name)

            # TODO add stuff to wandb or tensorboard
            wandb.log({"Loss": loss,
                       "Behaviour loss": loss_be,
                       "Reconstruction loss": loss_re})

    def save_checkpoint(self, state, dir=""):
        print("=> Saving checkpoint")
        path = "runs/" + dir
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
        torch.save(state, path + "/best.pt")

    def load_checkpoint(self, checkpoint, model):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()