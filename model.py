import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self,in_channels,out_channels, activation_function="elu"):
        super(Layer,self).__init__()
        self.activation_functions = {
            "elu" : nn.ELU(),
            "relu" : nn.ReLU(inplace=True),
            "leakyrelu" :nn.LeakyReLU(),
            "sigmoid" : nn.Sigmoid(),
            "tanh" : nn.Tanh(),
            "relu6" : nn.ReLU6()
           } 
        self.linear = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            self.activation_functions[activation_function]
        )
    def forward(self,x):
        return self.linear(x)

class Encoder(nn.Module):
    def __init__(
            self, activation_function="leakyrelu", exteroceptive=1080, encoder_features=[80,60], device='cuda:0'):
        super(Encoder,self).__init__()
        self.encoder = nn.ModuleList() 
        in_channels = exteroceptive
        for feature in encoder_features:
            self.encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

    def forward(self, x):
        
        for layer in self.encoder:
            print(layer)
            print("her")
            x = layer(x)
        
        return x

class Belief_Encoder(nn.Module):
    def __init__(
            self, proprioceptive=4, input_dim=60, hidden_dim=50,n_layers=2,activation_function="leakyrelu", network_features=[512,256,128], encoder_features=[80,60], device='cuda:0'):
        super(Belief_Encoder,self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        input_dim = proprioceptive+input_dim
        

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.gb = nn.ModuleList()
        self.ga = nn.ModuleList()
        gb_features = [64,64,60]
        ga_features = [64,64,60]

        in_channels = hidden_dim
        for feature in gb_features:
            self.gb.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        in_channels = hidden_dim
        for feature in ga_features:
            self.ga.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        self.ga.append(nn.Sigmoid())

        self.network = nn.ModuleList()  # MLP for network
        self.encoder = nn.ModuleList()  # Encoder with MLPs for heightmap
        self.input_dim = input_dim  # External information (Heightmap)
        self.num_proprioception = proprioceptive
        action_space=2
        # Create encoder for heightmap
        in_channels = self.input_dim
        for feature in encoder_features:
            self.encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        
        # Create MLP
        in_channels = self.num_proprioception + encoder_features[-1]
        for feature in network_features:
            self.network.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,action_space))
        self.network.append(nn.Tanh())

    def forward(self, p, l_e, h):
        print(h)
        # p = proprioceptive
        # e = exteroceptive
        # h = hidden state
        # x = input data, h = hidden state
        #p = p.squeeze()
        #l_e = l_e.squeeze()
        #print(torch.stack(p,l_e).shape) 
        x = torch.cat((p,l_e),dim=2)
        #x = self.encoder(x)
        out, h = self.gru(x, h)
        x_b = x_a = out

        for layer in self.gb:
            x_b = layer(x_b)
        for layer in self.ga:
            x_a = layer(x_a)

        x_a = l_e * x_a
        # TODO IMPLEMENT GATE
        belief = x_b + x_a

        return belief, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to('cpu')
        return hidden

class Belief_Decoder(nn.Module):
    def __init__(
            self, n_input=50, exteroceptive=1080, hidden_dim=50,n_layers=2,activation_function="leakyrelu", network_features=[512,256,128], encoder_features=[80,60], device='cuda:0'):
        super(Belief_Decoder,self).__init__()
        gate_features = [128,256,512,exteroceptive]
        decoder_features = [128,256,512,exteroceptive]
        self.n_input = n_input
        self.gate_encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
    

        in_channels = self.n_input
        for feature in gate_features:
            self.gate_encoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        self.gate_encoder.append(nn.Sigmoid())  

        in_channels = self.n_input
        for feature in decoder_features:
            self.decoder.append(Layer(in_channels, feature, activation_function))
            in_channels = feature
        

    def forward(self, e, h):
        gate = h[-1]
        decoded = h[-1]
        gate = gate.repeat(e.shape[1], 1, 1).permute(1,0,2)
        decoded = decoded.repeat(e.shape[1], 1, 1).permute(1,0,2)

        for layer in self.gate_encoder:
            gate = layer(gate)

        for layer in self.decoder:
            decoded = layer(decoded)
        print(e.shape)
        print(gate.shape)
        x = e*gate
        x = x + decoded
        return x


class MLP(nn.Module):
    def __init__(
            self, proprioceptive=4, belief_dim=60, activation_function="leakyrelu", network_features=[256,160,128], action_space=2, device='cuda:0'):
        super(MLP,self).__init__()
        self.network = nn.ModuleList()  # MLP for network


        in_channels = proprioceptive + belief_dim
        for feature in network_features:
            self.network.append(Layer(in_channels, feature, activation_function))
            in_channels = feature

        self.network.append(nn.Linear(in_channels,action_space))
        self.network.append(nn.Tanh())


    def forward(self, p, belief):
        
        x = torch.cat((p,belief),dim=2)

        for layer in self.network:
            x = layer(x)
        return x


class Student(nn.Module):
    def __init__(
            self, proprioceptive=4, exteroceptive=1913, n_actions=2, hidden_dim=50,n_layers=2,activation_function="leakyrelu", network_features=[512,256,128], encoder_features=[80,60], device='cuda:0'):
        super(Student,self).__init__()
        self.encoder = Encoder(exteroceptive=exteroceptive)
        self.belief_encoder = Belief_Encoder()
        self.belief_decoder = Belief_Decoder(exteroceptive=exteroceptive)
        self.MLP = MLP()
        self.n_p = proprioceptive
        self.n_e = exteroceptive
        self.n_a = n_actions


    def forward(self, x, h):
        n_a = self.n_a
        n_p = self.n_p
        reset = x[:,:, 0]
        actions = x[:,:,1:1+n_a]
        proprioceptive = x[:,:,1+n_a:1+n_a+n_p]
        exteroceptive = x[:,:,1+n_a+n_p:]
        print(x.shape)
        # n_p = self.n_p
        
        # p = x[:,:,0:n_p]        # Extract proprioceptive information  
        
        # e = x[:,:,n_p:1084]         # Extract exteroceptive information
        
        e_l = self.encoder(exteroceptive) # Pass exteroceptive information through encoder
        print("her")
        belief, h = self.belief_encoder(proprioceptive,e_l,h) # extract belief state
        
        estimated = self.belief_decoder(exteroceptive,h)
        

        action = self.MLP(proprioceptive,belief)

        return action, estimated