import torch
from torch import nn
from torch import optim
from model import Student
from tqdm import tqdm
from dataset import TeacherDataset
from torch.utils.data import DataLoader

class Trainer():
    def __init__(self):
        super(Trainer,self).__init__()
        self.LEARNING_RATE = 1e-4 # original 1e-4
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_EPOCHS = 10
        self.RUN_NAME = "TEST"
        self.BATCH_SIZE = 4

    def train_fn(self, train_loader, model, optimizer, loss_fn, scaler):
        loop = tqdm(train_loader)
        #TODO add metrics
        total_loss = 0
        
        h = model.belief_encoder.init_hidden(self.BATCH_SIZE).to(self.DEVICE)
        
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=self.DEVICE)

            #TODO format target to be in the correct format
            targets = targets.float().to(device=self.DEVICE)
            
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data,h)
                print(predictions)
                print(targets)
                loss_re = loss_fn["behaviour"](predictions, targets)
                loss_be = loss_fn["recontruction"](predictions, targets)
                loss = loss_be + 0.5 * loss_re

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # update tqdm loop
            loop.set_postfix(loss=loss.item())
            total_loss += loss.item()

        return total_loss

    def train(self):
        model = Student().to(self.DEVICE)
        loss_fn = {
            "behaviour":  nn.MSELoss(),
            "recontruction": nn.MSELoss()
        }
        optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
        scaler = torch.cuda.amp.GradScaler()
        train_ds = TeacherDataset("data/")
        train_loader = DataLoader(train_ds,batch_size=self.BATCH_SIZE,num_workers=4,pin_memory=True, shuffle=False)
        max_score = 0

        for epoch in range(0,self.NUM_EPOCHS):
            
            loss = self.train_fn(train_loader, model, optimizer, loss_fn, scaler)
            
             # Reset hidden units after each epoch

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            # TODO calculate metrics
            metrics = 0

            if metrics > max_score:
                print("Best model found => saving")
                max_score = metrics
                self.save_checkpoint(checkpoint,self.RUN_NAME)

            # TODO add stuff to wandb or tensorboard

    def save_checkpoint(self, state, filename="model/best.pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint, model):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()