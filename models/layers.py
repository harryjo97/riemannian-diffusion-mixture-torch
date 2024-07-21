import torch
import numpy as np

def get_activation(act, **kwargs):
    if act=='swish':
        return torch.nn.SiLU()
    elif act=='sin':
        return Sin_Act()
    else:
        raise NotImplementedError(f'Activation {act} not implemented.')


class Sin_Act(torch.nn.Module):
	def __init__(self):
		super(Sin_Act, self).__init__()

	def forward(self, x):
		return torch.sin(x)



class MLP(torch.nn.Module):
    def __init__(self, num_layers, in_dim, hid_dim, out_dim, act, bias=True):
        super().__init__()
        hidden_shapes = [hid_dim] * num_layers

        self.linears = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        for n in range(len(hidden_shapes)+1):
            if n==0:
                self.linears.append(torch.nn.Linear(in_dim, hidden_shapes[0], bias=bias))
            elif n==len(hidden_shapes):
                self.linears.append(torch.nn.Linear(hidden_shapes[-1], out_dim, bias=bias))
            else:
                self.linears.append(torch.nn.Linear(hidden_shapes[n-1], hidden_shapes[n], bias=bias))
            
        for n in range(len(hidden_shapes)):
            if n  < len(hidden_shapes)-1:
                self.acts.append(get_activation(act, in_dim=hidden_shapes[n], out_dim=hidden_shapes[n+1]))
            else:
                self.acts.append(get_activation(act, in_dim=hidden_shapes[n], out_dim=out_dim))
            
    def forward(self, x):
        for _, linear in enumerate(self.linears):
            x = linear(x)
            if _ < len(self.linears)-1:
                x = self.acts[_](x)
        return x