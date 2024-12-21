import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResNet4D(nn.Module):
    def __init__(self, num_inputs=4, num_outputs=4, num_hidden_layers=4, num_neurons=50):
        super(ResNet4D, self).__init__()
        self.input_layer = nn.Linear(num_inputs, num_neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(num_neurons, num_neurons) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(num_neurons, num_outputs)
        self.activation = Swish()

    def forward(self, x, y, z, t):
        x_norm = (x - 0.5) / 0.2887
        y_norm = (y - 0.5) / 0.2887
        z_norm = (z - 0.5) / 0.2887
        t_norm = (t - 0.5) / 0.2887

        inputs = torch.cat([x_norm, y_norm, z_norm, t_norm], dim=1)
        out = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            residual = out
            out = self.activation(layer(out))
            out = out + residual  # Residual connection
        out = self.output_layer(out)
        return out
