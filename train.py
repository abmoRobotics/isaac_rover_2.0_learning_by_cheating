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
    def __init__(self, cfg):
        super(Trainer,self).__init__()
        self.cfg = cfg
        cfg_hyperparameters = cfg["learning"]
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.LEARNING_RATE = cfg_hyperparameters["learning_rate"] # original 1e-4
        self.NUM_EPOCHS = cfg_hyperparameters["epochs"]
        self.BATCH_SIZE = cfg_hyperparameters["batch_size"]
        
        self.RUN_NAME = "TEST"
        self.wandb_group = "test-group"
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.wandb_name = f"test-run_{time_str}"


    def train_fn(self, train_loader, model, optimizer, loss_fn, scaler):
        loop = tqdm(train_loader)
        #TODO add metrics
        total_loss = 0
        total_be_loss = 0
        total_re_loss = 0
        total_loss_benchmark = 0
        
        
        for batch_idx, (data, targets_ac, targets_ex) in enumerate(loop):
            data = data.to(device=self.DEVICE)
            h = model.belief_encoder.init_hidden(self.BATCH_SIZE).to(self.DEVICE)
            #TODO format target to be in the correct format
            targets_ac = targets_ac.float().to(device=self.DEVICE)
            targets_ex = targets_ex.float().to(device=self.DEVICE)

            actions = torch.zeros(self.BATCH_SIZE,data.shape[1], 2,device='cuda:0')
            predictions = torch.zeros(self.BATCH_SIZE,data.shape[1], data.shape[2]-7,device='cuda:0')
            # print(predictions.shape)
            # print(targets_ex.shape)
            # forward
            

            with torch.cuda.amp.autocast():
                # print(data.shape)
                # print(data[:,1].unsqueeze(1).shape)

                #data = torch.cat([data[0:2], data[4:]])
                #data[:,:,2:4]=0.0
                actions, predictions, h = model(data,h)

                # for i in range(data.shape[1]-1):
                #     # if i%10 ==0:
                #     #     print(i)
                #     a,p,h = model(data[:,i].unsqueeze(1),h)
                #     actions[:,i,:]  = a.squeeze()
                #     predictions[:,i,:] = p.squeeze() # [num_robots, timestep, observations]'
                #     #print(i)
                #     # print(p.shape)
                #     # print(predictions.shape)
                #     data[:,i+1,2:4] = actions[:,i]
          
                loss_be = loss_fn["behaviour"](actions, targets_ac)
                #print(actions[0,50:70])
                #print(targets_ac[0,50:70])
                loss_re = loss_fn["recontruction"](predictions, targets_ex)
                #loss_re = my_loss(predictions,targets_ex)*100
                loss_benchmark = loss_fn["recontruction"](data[:,:,7:],targets_ex)
                loss = 1.0 * loss_be + (0.0000005 * loss_re)
                #torch.cuda.empty_cache() 
                
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # update tqdm loop
            loop.set_postfix(loss=loss.item())
            if batch_idx == 15:
                torch.save(predictions[0,100,:].detach(),"predictions.pt")
                torch.save(data[0,100,7:].detach(),"input.pt")
                torch.save(targets_ex[0,100,:].detach(),"targets.pt")
                print("noisey", data[0,100,67:77].detach())
                print("Predictions", predictions[0,100,60:70].detach())
                print("GT: ", targets_ex[0,100,60:70].detach()) # [num_robots, timestep, observations]
                print("Pred Sum", torch.abs(predictions[0,100]).sum())
            total_loss += loss.item()
            total_be_loss += loss_be
            total_re_loss += loss_re
            total_loss_benchmark += loss_benchmark
    
        return total_loss, total_be_loss, total_re_loss, total_loss_benchmark

    def train(self):
        wandb.init(project='isaac-rover-2.0-learning-by-cheating', sync_tensorboard=True,name=self.wandb_name,group=self.wandb_group, entity="aalborg-university")
        train_ds = TeacherDataset("data/")
        train_loader = DataLoader(train_ds,batch_size=self.BATCH_SIZE,num_workers=4,pin_memory=True, shuffle=False)
        
        model = Student(info=train_ds.get_info(), cfg=self.cfg, teacher="teacher_model/agent_219000.pt").to(self.DEVICE)
        loss_fn = {
            "behaviour":     nn.MSELoss(reduction="mean"),
            "recontruction": nn.MSELoss(reduction="mean")
        }
        # Define paramters for optimizer
        parameters=[]
        #parameters.extend(model.encoder.parameters())
        parameters.extend(model.belief_encoder.parameters())
        parameters.extend(model.belief_decoder.parameters())
        # parameters.extend(model.encoder1.parameters())
        # parameters.extend(model.encoder2.parameters())
        # Set MLP.parameters() to false, to avoid accumulating unessecary gradients.
        for param in model.MLP.parameters():
            param.requires_grad = False
        for param in model.encoder1.parameters():
            param.requires_grad = False
        for param in model.encoder2.parameters():
            param.requires_grad = False
        # for param in model.belief_decoder.parameters():
        #     param.requires_grad = False
        # Define optimzer
        optimizer = optim.Adam(parameters, lr=self.LEARNING_RATE)
        #optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
        # Use automatic mixed precision
        scaler = torch.cuda.amp.GradScaler()

        # Metric for best policy
        best = float('inf')

        for epoch in range(0,self.NUM_EPOCHS):
            self.epoch = epoch
            loss, loss_be, loss_re, loss_benchmark = self.train_fn(train_loader, model, optimizer, loss_fn, scaler)
            # print(loss_be/len(train_loader))
            # print(loss_re/len(train_loader))
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
            wandb.log({"Loss": loss/len(train_loader),
                       "Behaviour loss": loss_be/len(train_loader),
                       "Reconstruction loss": loss_re/len(train_loader),
                       "Benchmark loss": loss_benchmark/len(train_loader)})

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


def cfg_fn():
    cfg = {
        "info":{
            "reset":            0,
            "actions":          0,
            "proprioceptive":   0,
            "exteroceptive":    0,
        },
        "learning":{
            "learning_rate": 1e-4,
            "epochs": 500,
            "batch_size": 8,
        },
        "encoder":{
            "activation_function": "leakyrelu",
            "encoder_features": [80,60]},

        "belief_encoder": {
            "hidden_dim":       300,
            "n_layers":         2,
            "activation_function":  "leakyrelu",
            "gb_features": [64,64,120],
            "ga_features": [64,64,120]},

        "belief_decoder": {
            "activation_function": "leakyrelu",
            "gate_features":    [128,256,512],
            "decoder_features": [128,256,512]
        },
        "mlp":{"activation_function": "leakyrelu",
            "network_features": [256,160,128]},
            }

    return cfg

def train():
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_group = f"test-group_{time_str}"
    #wandb_group = "test-group"
    wandb_name = f"test-run_{time_str}"

    wandb.init(project='isaac-rover-2.0-learning-by-cheating', sync_tensorboard=True, name=wandb_name, group=wandb_group, entity="aalborg-university")
    cfg = cfg_fn()
    sweep=False
    if sweep:
        #print(wandb.config)
        cfg["belief_encoder"]["gb_features"] = wandb.config.gb_features
        cfg["belief_encoder"]["ga_features"] = wandb.config.gb_features
        cfg["belief_decoder"]["gate_features"] = wandb.config.gate_features
        cfg["belief_decoder"]["decoder_features"] = wandb.config.gate_features
        cfg["learning"]["learning_rate"] = wandb.config.lr
        cfg["learning"]["batch_size"] = wandb.config.batch_size
    trainer = Trainer(cfg)
    trainer.train()

def my_loss(output: torch.Tensor, target: torch.Tensor):
    x = (output- target)# 1 - output.square()#(torch.square(output))
    x = x.square()
    x = x.div(0.05) #-torch.div(x,2*0.04) # x.div(2*0.04)
    x = (x).exp() #torch.exp(x)
    loss = torch.mean(x)
    return loss

if __name__ == "__main__":
    sweep = False
    
    if not sweep:
        train()
    else:
        n_sweeps = 30
        sweep_configuration = {
            'method': 'bayes',
            'name': 'sweep',
            'metric': {'goal': 'minimize', 'name': 'Reconstruction loss'},
            'parameters': 
            {
                'batch_size': {'values': [4, 8, 16, 32]},
                'lr': {'max': 0.003, 'min': 0.00003},
                'gb_features': {'values': [[32,32,60], [64,64,60], [128,128,60]]},
                'gate_features': {'values': [[128,256], [256,512,1024], [128,256,512,1024],[128,256,512]]},

                
            }
        }
        # Initialize sweep by passing in config. (Optional) Provide a name of the project.
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='isaac-rover-2.0-learning-by-cheating')
        wandb.agent(sweep_id, function=train, count=n_sweeps)