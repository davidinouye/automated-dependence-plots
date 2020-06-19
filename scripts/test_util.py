import torch

def f_relu(x):
    return torch.nn.ReLU()(x[0])

def f_const(phi_val, const = 0):
    return torch.tensor(const).float()

def f_linear(x):
    return x[0]

def f_prod(x):#x^y example
    return torch.prod(x) 

def f_x_square(x):#x^2 example
    return x[0] * x[0]

def f_sin_x(x):
    return torch.sin(x[0])

def f_bryon_2(x):
    return torch.sin(torch.sqrt(torch.pow(x[0], 2) + torch.pow(x[1], 2))/(-2 + torch.atan(torch.tensor([x[0]/x[1]]))))

def f_bryon_0(x):
    return torch.sin(torch.sqrt(torch.pow(x[0], 2) + torch.pow(x[1], 2))/( torch.atan(torch.tensor([x[0]/x[1]]))))
