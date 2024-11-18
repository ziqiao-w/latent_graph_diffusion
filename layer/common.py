import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_unit_sphere(n_samples, n_nodes, max_pos):
    res = torch.zeros(n_samples, n_nodes, max_pos)
    m = min(n_nodes - 1, max_pos)
    x = torch.randn(n_samples, n_nodes, m)
    y = x.norm(p=2, dim=1, keepdim=True)
    x /= y

    sign_flip = torch.rand(m)
    sign_flip[sign_flip>=0.5] = 1.0
    sign_flip[sign_flip<0.5] = -1.0

    x = x * sign_flip.unsqueeze(0).unsqueeze(0)
    res[:, :, :m].copy_(x)
    
    return res


"""
    MLP Layer used after graph vector representation
"""

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y
    

class FFN_SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 128,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


if __name__ == "__main__":
    print(sample_unit_sphere(2, 4, 8))
