import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResNet2D(nn.Module):
    def __init__(self, num_inputs=2, num_outputs=1, num_hidden_layers=4, num_neurons=50):
        super(ResNet2D, self).__init__()
        self.input_layer = nn.Linear(num_inputs, num_neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(num_neurons, num_neurons) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(num_neurons, num_outputs)
        self.activation = Swish()

    def forward(self, x, t):
        x_norm = (x - 0.5) / 0.2887
        t_norm = (t - 0.5) / 0.2887

        inputs = torch.cat([x_norm, t_norm], dim=1)
        out = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            residual = out
            out = self.activation(layer(out))
            out = out + residual  
        out = self.output_layer(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, num_inputs=3, num_outputs=1, num_hidden_layers=4, num_neurons=50):
        super(ResNet3D, self).__init__()
        self.input_layer = nn.Linear(num_inputs, num_neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(num_neurons, num_neurons) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(num_neurons, num_outputs)
        self.activation = Swish()

    def forward(self, x, y, t):
        # Normalize inputs
        x_norm = (x - 0.5) / 0.2887
        y_norm = (y - 0.5) / 0.2887
        t_norm = (t - 0.5) / 0.2887

        inputs = torch.cat([x_norm, y_norm, t_norm], dim=1)
        out = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            residual = out
            out = self.activation(layer(out))
            out = out + residual  # Residual connection
        out = self.output_layer(out)
        return out

class ResNet4D(nn.Module):
    def __init__(self, num_inputs=4, num_outputs=1, num_hidden_layers=4, num_neurons=50):
        super(ResNet4D, self).__init__()
        self.input_layer = nn.Linear(num_inputs, num_neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(num_neurons, num_neurons) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(num_neurons, num_outputs)
        self.activation = Swish()

    def forward(self, x, y, z, t):
        # Normalize inputs
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


class Solver2D:
    def __init__(self, pde_residual, initial_condition_function,
                 domain_bounds, model_params, training_params):
        """
        Initializes the solver.

        Parameters:
        - pde_residual: Function defining the PDE residual.
        - initial_condition_function: Function defining u0(x) for reparameterization.
        - domain_bounds: Dictionary with 'x', 't' keys and (min, max) tuples.
        - model_params: Dictionary with model parameters (e.g., number of layers).
        - training_params: Dictionary with training parameters (e.g., epochs, batch size).
        """
        self.pde_residual = pde_residual
        self.initial_condition_function = initial_condition_function
        self.domain_bounds = domain_bounds
        self.model_params = model_params
        self.training_params = training_params

        # Initialize the neural network model
        self.model = ResNet2D(num_inputs=2,
                              num_outputs=1,
                              num_hidden_layers=model_params.get('num_hidden_layers', 4),
                              num_neurons=model_params.get('num_neurons', 50)).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=training_params.get('learning_rate', 1e-3))

        self.pde_loss_history = []
        self.bc_loss_history = []
        self.total_loss_history = []

        # Adaptive loss weighting
        self.w_pde = 1.0
        self.w_bc = 1.0

    # Defining the spatial modulation function ξ(x)
    def spatial_modulation(self, x):
        x_min, x_max = self.domain_bounds['x']
        return (x - x_min) * (x_max - x)

    # Defining the temporal modulation function τ(t)
    def temporal_modulation(self, t):
        t_min, t_max = self.domain_bounds['t']
        return (t - t_min)

    def reparameterize_solution(self, x, t):
        x = x.view(-1, 1)
        t = t.view(-1, 1)

        N = self.model(x, t)

        u0 = self.initial_condition_function(x)
        xi = self.spatial_modulation(x)
        tau = self.temporal_modulation(t)

        u = u0 + tau * xi * N
        return u

    def generate_training_data(self):
        # Generate collocation points in the interior of the domain
        N_f = self.training_params.get('N_f', 10000)
        x_f = torch.rand(N_f, 1, requires_grad=True).to(device) * (self.domain_bounds['x'][1] - self.domain_bounds['x'][0]) + self.domain_bounds['x'][0]
        t_f = torch.rand(N_f, 1, requires_grad=True).to(device) * (self.domain_bounds['t'][1] - self.domain_bounds['t'][0]) + self.domain_bounds['t'][0]

        # Generate boundary points
        N_b = self.training_params.get('N_b', 2000)
        # Boundary at x = x_min and x = x_max
        x_b0 = torch.full((N_b, 1), self.domain_bounds['x'][0], requires_grad=True).to(device)
        x_b1 = torch.full((N_b, 1), self.domain_bounds['x'][1], requires_grad=True).to(device)
        t_b = torch.rand(2 * N_b, 1, requires_grad=True).to(device) * (self.domain_bounds['t'][1] - self.domain_bounds['t'][0]) + self.domain_bounds['t'][0]
        x_b = torch.cat([x_b0, x_b1], dim=0)

        return x_f, t_f, x_b, t_b

    def loss_function(self, x_f, t_f, x_b, t_b):
        # PDE residuals
        f_pred = self.pde_residual(x_f, t_f, self.reparameterize_solution)
        mse_pde = torch.mean(f_pred ** 2)

        # Boundary conditions at x boundaries
        u_b = self.reparameterize_solution(x_b, t_b)
        mse_bc = torch.mean(u_b ** 2)

        # Total loss with adaptive weighting
        loss = self.w_pde * mse_pde + self.w_bc * mse_bc

        # Record individual losses
        self.pde_loss_history.append(mse_pde.item())
        self.bc_loss_history.append(mse_bc.item())
        self.total_loss_history.append(loss.item())

        return loss

    def train(self):
        num_epochs = self.training_params.get('num_epochs', 5000)
        patience = self.training_params.get('patience', 500)
        min_delta = self.training_params.get('min_delta', 1e-6)
        adjust_interval = self.training_params.get('adjust_interval', 100)

        best_loss = np.inf
        epochs_no_improve = 0

        start_time = time.time()

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            # Move data generation inside the loop
            x_f, t_f, x_b, t_b = self.generate_training_data()

            loss = self.loss_function(x_f, t_f, x_b, t_b)

            loss.backward()
            self.optimizer.step()

            current_loss = loss.item()

            # Adaptive loss weighting
            if epoch % adjust_interval == 0 and epoch != 0:
                avg_pde_loss = np.mean(self.pde_loss_history[-adjust_interval:])
                avg_bc_loss = np.mean(self.bc_loss_history[-adjust_interval:])

                # Adjust weights inversely proportional to losses
                total_loss = avg_pde_loss + avg_bc_loss + 1e-8  
                self.w_pde = total_loss / (2 * avg_pde_loss + 1e-8)
                self.w_bc = total_loss / (2 * avg_bc_loss + 1e-8)

                weight_sum = self.w_pde + self.w_bc
                self.w_pde /= weight_sum
                self.w_bc /= weight_sum

            if best_loss - current_loss > min_delta:
                best_loss = current_loss
                epochs_no_improve = 0
                # torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if epoch % 100 == 0:
                elapsed_time = time.time() - start_time
                print('Epoch: %d, Loss: %.5e, w_pde: %.3f, w_bc: %.3f, Time Elapsed: %.2f s' %
                      (epoch, loss.item(), self.w_pde, self.w_bc, elapsed_time))

    def predict(self, x, t):
        x = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
        t = torch.tensor(t, dtype=torch.float32).view(-1, 1).to(device)

        with torch.no_grad():
            u_pred = self.reparameterize_solution(x, t)
        return u_pred.cpu().numpy()

    def plot_results(self, time_steps):
        x_plot = np.linspace(self.domain_bounds['x'][0], self.domain_bounds['x'][1], 100)
        X, T = np.meshgrid(x_plot, time_steps)

        X_flat = X.flatten()
        T_flat = T.flatten()

        u_pred = self.predict(X_flat, T_flat)
        U = u_pred.reshape(X.shape)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, T, U, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x, t)')
        plt.show()

    def plot_loss_history(self):
        plt.figure()
        plt.plot(self.total_loss_history)
        #plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.ylim(-0.05, max(self.total_loss_history))
        plt.title('Training Loss History')
        plt.show()


class Solver3D:
    def __init__(self, pde_residual, initial_condition_function,
                 domain_bounds, model_params, training_params):
        """
        Initializes the solver.

        Parameters:
        - pde_residual: Function defining the PDE residual.
        - initial_condition_function: Function defining u0(x, y) for reparameterization.
        - domain_bounds: Dictionary with 'x', 'y', 't' keys and (min, max) tuples.
        - model_params: Dictionary with model parameters (e.g., number of layers).
        - training_params: Dictionary with training parameters (e.g., epochs, batch size).
        """
        self.pde_residual = pde_residual
        self.initial_condition_function = initial_condition_function
        self.domain_bounds = domain_bounds
        self.model_params = model_params
        self.training_params = training_params

        # Initialize the neural network model
        self.model = ResNet3D(num_inputs=3,
                              num_outputs=1,
                              num_hidden_layers=model_params.get('num_hidden_layers', 4),
                              num_neurons=model_params.get('num_neurons', 50)).to(device)

        # Set up the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=training_params.get('learning_rate', 1e-3))

        # Loss histories
        self.pde_loss_history = []
        self.bc_loss_history = []
        self.total_loss_history = []

        # Adaptive loss weighting
        self.w_pde = 1.0
        self.w_bc = 1.0

    # Define the spatial modulation function ξ(x, y)
    def spatial_modulation(self, x, y):
        x_min, x_max = self.domain_bounds['x']
        y_min, y_max = self.domain_bounds['y']
        return (x - x_min) * (x_max - x) * (y - y_min) * (y_max - y)

    # Define the temporal modulation function τ(t)
    def temporal_modulation(self, t):
        t_min, t_max = self.domain_bounds['t']
        return (t - t_min)

    # Define the reparameterization function u(x, y, t)
    def reparameterize_solution(self, x, y, t):
        # Reshape x, y, t to column vectors
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        t = t.view(-1, 1)

        # Neural network output
        N = self.model(x, y, t)

        # Initial condition u0(x, y)
        u0 = self.initial_condition_function(x, y)

        # Modulation functions
        xi = self.spatial_modulation(x, y)
        tau = self.temporal_modulation(t)

        # Reparameterized solution u(x, y, t)
        u = u0 + tau * xi * N
        return u

    def generate_training_data(self):
        # Generate collocation points in the interior of the domain
        N_f = self.training_params.get('N_f', 10000)
        x_f = torch.rand(N_f, 1, requires_grad=True).to(device) * (self.domain_bounds['x'][1] - self.domain_bounds['x'][0]) + self.domain_bounds['x'][0]
        y_f = torch.rand(N_f, 1, requires_grad=True).to(device) * (self.domain_bounds['y'][1] - self.domain_bounds['y'][0]) + self.domain_bounds['y'][0]
        t_f = torch.rand(N_f, 1, requires_grad=True).to(device) * (self.domain_bounds['t'][1] - self.domain_bounds['t'][0]) + self.domain_bounds['t'][0]

        # Generate boundary points
        N_b = self.training_params.get('N_b', 2000)
        # Boundary at x = x_min and x = x_max
        x_b0 = torch.full((N_b, 1), self.domain_bounds['x'][0], requires_grad=True).to(device)
        x_b1 = torch.full((N_b, 1), self.domain_bounds['x'][1], requires_grad=True).to(device)
        y_bx = torch.rand(2 * N_b, 1, requires_grad=True).to(device) * (self.domain_bounds['y'][1] - self.domain_bounds['y'][0]) + self.domain_bounds['y'][0]
        t_bx = torch.rand(2 * N_b, 1, requires_grad=True).to(device) * (self.domain_bounds['t'][1] - self.domain_bounds['t'][0]) + self.domain_bounds['t'][0]
        x_bx = torch.cat([x_b0, x_b1], dim=0)

        # Boundary at y = y_min and y = y_max
        y_b0 = torch.full((N_b, 1), self.domain_bounds['y'][0], requires_grad=True).to(device)
        y_b1 = torch.full((N_b, 1), self.domain_bounds['y'][1], requires_grad=True).to(device)
        x_by = torch.rand(2 * N_b, 1, requires_grad=True).to(device) * (self.domain_bounds['x'][1] - self.domain_bounds['x'][0]) + self.domain_bounds['x'][0]
        t_by = torch.rand(2 * N_b, 1, requires_grad=True).to(device) * (self.domain_bounds['t'][1] - self.domain_bounds['t'][0]) + self.domain_bounds['t'][0]
        y_by = torch.cat([y_b0, y_b1], dim=0)

        return x_f, y_f, t_f, x_bx, y_bx, t_bx, x_by, y_by, t_by

    def loss_function(self, x_f, y_f, t_f, x_bx, y_bx, t_bx, x_by, y_by, t_by):
        # PDE residuals
        f_pred = self.pde_residual(x_f, y_f, t_f, self.reparameterize_solution)
        mse_pde = torch.mean(f_pred ** 2)

        # Boundary conditions at x boundaries
        u_bx = self.reparameterize_solution(x_bx, y_bx, t_bx)
        mse_bx = torch.mean(u_bx ** 2)

        # Boundary conditions at y boundaries
        u_by = self.reparameterize_solution(x_by, y_by, t_by)
        mse_by = torch.mean(u_by ** 2)

        mse_bc = mse_bx + mse_by

        # Total loss with adaptive weighting
        loss = self.w_pde * mse_pde + self.w_bc * mse_bc

        # Record individual losses
        self.pde_loss_history.append(mse_pde.item())
        self.bc_loss_history.append(mse_bc.item())
        self.total_loss_history.append(loss.item())

        return loss

    def train(self):
        num_epochs = self.training_params.get('num_epochs', 5000)
        patience = self.training_params.get('patience', 500)
        min_delta = self.training_params.get('min_delta', 1e-6)
        adjust_interval = self.training_params.get('adjust_interval', 100)

        best_loss = np.inf
        epochs_no_improve = 0

        start_time = time.time()

        with tqdm(total=num_epochs, desc="Training Progress") as pbar:
            for epoch in range(num_epochs):
                self.optimizer.zero_grad()

                # Move data generation inside the loop
                x_f, y_f, t_f, x_bx, y_bx, t_bx, x_by, y_by, t_by = self.generate_training_data()

                loss = self.loss_function(x_f, y_f, t_f, x_bx, y_bx, t_bx, x_by, y_by, t_by)

                loss.backward()
                self.optimizer.step()

                current_loss = loss.item()

                # Adaptive loss weighting
                if epoch % adjust_interval == 0 and epoch != 0:
                    # Compute average losses over the interval
                    avg_pde_loss = np.mean(self.pde_loss_history[-adjust_interval:])
                    avg_bc_loss = np.mean(self.bc_loss_history[-adjust_interval:])

                    # Adjust weights inversely proportional to losses
                    total_loss = avg_pde_loss + avg_bc_loss + 1e-8  # Add small epsilon to avoid division by zero
                    self.w_pde = total_loss / (2 * avg_pde_loss + 1e-8)
                    self.w_bc = total_loss / (2 * avg_bc_loss + 1e-8)

                    # Normalize weights to sum to 1
                    weight_sum = self.w_pde + self.w_bc
                    self.w_pde /= weight_sum
                    self.w_bc /= weight_sum

                # Early stopping check
                if best_loss - current_loss > min_delta:
                    best_loss = current_loss
                    epochs_no_improve = 0
                    # Optionally, save the model
                    # torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

                # Update tqdm progress bar
                pbar.set_postfix({
                    'Epoch': epoch,
                    'Loss': f'{loss.item():.5e}',
                    'w_pde': f'{self.w_pde:.3f}',
                    'w_bc': f'{self.w_bc:.3f}',
                    'Elapsed Time': f'{time.time() - start_time:.2f}s'
                })
                pbar.update(1)

    def predict(self, x, y, t):
        x = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
        t = torch.tensor(t, dtype=torch.float32).view(-1, 1).to(device)

        with torch.no_grad():
            u_pred = self.reparameterize_solution(x, y, t)
        return u_pred.cpu().numpy()

    def plot_results(self, time_steps):
        x_plot = np.linspace(self.domain_bounds['x'][0], self.domain_bounds['x'][1], 50)
        y_plot = np.linspace(self.domain_bounds['y'][0], self.domain_bounds['y'][1], 50)
        X, Y = np.meshgrid(x_plot, y_plot)

        fig = plt.figure(figsize=(15, 10))

        for idx, t_val in enumerate(time_steps):
            # Flatten the grid arrays
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            T_flat = t_val * np.ones_like(X_flat)

            # Predict the solution
            u_pred = self.predict(X_flat, Y_flat, T_flat)
            U = u_pred.reshape(X.shape)

            # Plot the solution u(x, y, t) at the current time slice
            ax = fig.add_subplot(3, 3, idx + 1, projection='3d')
            ax.plot_surface(X, Y, U, cmap='viridis')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u(x, y, t)')
            ax.set_title(f't = {t_val}')
            plt.tight_layout()

        plt.show()

    def plot_loss_history(self):
        plt.figure()
        plt.plot(self.total_loss_history)
        #plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.ylim(-1.0, max(self.total_loss_history))
        plt.title('Training Loss History')
        plt.show()


class Solver4D:
    def __init__(self, pde_residual, initial_condition_function,
                 domain_bounds, model_params, training_params):
        """
        Initializes the solver.

        Parameters:
        - pde_residual: Function defining the PDE residual.
        - initial_condition_function: Function defining u0(x, y, z) for reparameterization.
        - domain_bounds: Dictionary with 'x', 'y', 'z', 't' keys and (min, max) tuples.
        - model_params: Dictionary with model parameters (e.g., number of layers).
        - training_params: Dictionary with training parameters (e.g., epochs, batch size).
        """
        self.pde_residual = pde_residual
        self.initial_condition_function = initial_condition_function
        self.domain_bounds = domain_bounds
        self.model_params = model_params
        self.training_params = training_params

        # Initialize the neural network model
        self.model = ResNet4D(num_inputs=4,
                              num_outputs=1,
                              num_hidden_layers=model_params.get('num_hidden_layers', 4),
                              num_neurons=model_params.get('num_neurons', 50)).to(device)

        # Set up the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=training_params.get('learning_rate', 1e-3))

        # Loss histories
        self.pde_loss_history = []
        self.bc_loss_history = []
        self.total_loss_history = []

        # Adaptive loss weighting
        self.w_pde = 1.0
        self.w_bc = 1.0

    # Define the spatial modulation function ξ(x, y, z)
    def spatial_modulation(self, x, y, z):
        x_min, x_max = self.domain_bounds['x']
        y_min, y_max = self.domain_bounds['y']
        z_min, z_max = self.domain_bounds['z']
        return (x - x_min) * (x_max - x) * (y - y_min) * (y_max - y) * (z - z_min) * (z_max - z)

    # Define the temporal modulation function τ(t)
    def temporal_modulation(self, t):
        t_min, t_max = self.domain_bounds['t']
        return (t - t_min)

    # Define the reparameterization function u(x, y, z, t)
    def reparameterize_solution(self, x, y, z, t):
        # Reshape x, y, z, t to column vectors
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        z = z.view(-1, 1)
        t = t.view(-1, 1)

        # Neural network output
        N = self.model(x, y, z, t)

        # Initial condition u0(x, y, z)
        u0 = self.initial_condition_function(x, y, z)

        # Modulation functions
        xi = self.spatial_modulation(x, y, z)
        tau = self.temporal_modulation(t)

        # Reparameterized solution u(x, y, z, t)
        u = u0 + tau * xi * N
        return u

    def generate_training_data(self):
        # Generate collocation points in the interior of the domain
        N_f = self.training_params.get('N_f', 10000)
        x_f = torch.rand(N_f, 1, requires_grad=True).to(device) * (self.domain_bounds['x'][1] - self.domain_bounds['x'][0]) + self.domain_bounds['x'][0]
        y_f = torch.rand(N_f, 1, requires_grad=True).to(device) * (self.domain_bounds['y'][1] - self.domain_bounds['y'][0]) + self.domain_bounds['y'][0]
        z_f = torch.rand(N_f, 1, requires_grad=True).to(device) * (self.domain_bounds['z'][1] - self.domain_bounds['z'][0]) + self.domain_bounds['z'][0]
        t_f = torch.rand(N_f, 1, requires_grad=True).to(device) * (self.domain_bounds['t'][1] - self.domain_bounds['t'][0]) + self.domain_bounds['t'][0]

        # Generate boundary points
        N_b = self.training_params.get('N_b', 2000)
        # Boundary at x = x_min and x = x_max
        x_b0 = torch.full((N_b, 1), self.domain_bounds['x'][0], requires_grad=True).to(device)
        x_b1 = torch.full((N_b, 1), self.domain_bounds['x'][1], requires_grad=True).to(device)
        y_bx = torch.rand(2 * N_b, 1, requires_grad=True).to(device) * (self.domain_bounds['y'][1] - self.domain_bounds['y'][0]) + self.domain_bounds['y'][0]
        z_bx = torch.rand(2 * N_b, 1, requires_grad=True).to(device) * (self.domain_bounds['z'][1] - self.domain_bounds['z'][0]) + self.domain_bounds['z'][0]
        t_bx = torch.rand(2 * N_b, 1, requires_grad=True).to(device) * (self.domain_bounds['t'][1] - self.domain_bounds['t'][0]) + self.domain_bounds['t'][0]
        x_bx = torch.cat([x_b0, x_b1], dim=0)

        # Boundary at y = y_min and y = y_max
        y_b0 = torch.full((N_b, 1), self.domain_bounds['y'][0], requires_grad=True).to(device)
        y_b1 = torch.full((N_b, 1), self.domain_bounds['y'][1], requires_grad=True).to(device)
        x_by = torch.rand(2 * N_b, 1, requires_grad=True).to(device) * (self.domain_bounds['x'][1] - self.domain_bounds['x'][0]) + self.domain_bounds['x'][0]
        z_by = torch.rand(2 * N_b, 1, requires_grad=True).to(device) * (self.domain_bounds['z'][1] - self.domain_bounds['z'][0]) + self.domain_bounds['z'][0]
        t_by = torch.rand(2 * N_b, 1, requires_grad=True).to(device) * (self.domain_bounds['t'][1] - self.domain_bounds['t'][0]) + self.domain_bounds['t'][0]
        y_by = torch.cat([y_b0, y_b1], dim=0)

        # Boundary at z = z_min and z = z_max
        z_b0 = torch.full((N_b, 1), self.domain_bounds['z'][0], requires_grad=True).to(device)
        z_b1 = torch.full((N_b, 1), self.domain_bounds['z'][1], requires_grad=True).to(device)
        x_bz = torch.rand(2 * N_b, 1, requires_grad=True).to(device) * (self.domain_bounds['x'][1] - self.domain_bounds['x'][0]) + self.domain_bounds['x'][0]
        y_bz = torch.rand(2 * N_b, 1, requires_grad=True).to(device) * (self.domain_bounds['y'][1] - self.domain_bounds['y'][0]) + self.domain_bounds['y'][0]
        t_bz = torch.rand(2 * N_b, 1, requires_grad=True).to(device) * (self.domain_bounds['t'][1] - self.domain_bounds['t'][0]) + self.domain_bounds['t'][0]
        z_bz = torch.cat([z_b0, z_b1], dim=0)

        return x_f, y_f, z_f, t_f, x_bx, y_bx, z_bx, t_bx, x_by, y_by, z_by, t_by, x_bz, y_bz, z_bz, t_bz

    def loss_function(self, x_f, y_f, z_f, t_f,
                      x_bx, y_bx, z_bx, t_bx,
                      x_by, y_by, z_by, t_by,
                      x_bz, y_bz, z_bz, t_bz):
        # PDE residuals
        f_pred = self.pde_residual(x_f, y_f, z_f, t_f, self.reparameterize_solution)
        mse_pde = torch.mean(f_pred ** 2)

        # Boundary conditions at x boundaries
        u_bx = self.reparameterize_solution(x_bx, y_bx, z_bx, t_bx)
        mse_bx = torch.mean(u_bx ** 2)

        # Boundary conditions at y boundaries
        u_by = self.reparameterize_solution(x_by, y_by, z_by, t_by)
        mse_by = torch.mean(u_by ** 2)

        # Boundary conditions at z boundaries
        u_bz = self.reparameterize_solution(x_bz, y_bz, z_bz, t_bz)
        mse_bz = torch.mean(u_bz ** 2)

        mse_bc = mse_bx + mse_by + mse_bz

        # Total loss with adaptive weighting
        loss = self.w_pde * mse_pde + self.w_bc * mse_bc

        # Record individual losses
        self.pde_loss_history.append(mse_pde.item())
        self.bc_loss_history.append(mse_bc.item())
        self.total_loss_history.append(loss.item())

        return loss

    def train(self):
        num_epochs = self.training_params.get('num_epochs', 5000)
        patience = self.training_params.get('patience', 500)
        min_delta = self.training_params.get('min_delta', 1e-6)
        adjust_interval = self.training_params.get('adjust_interval', 100)

        best_loss = np.inf
        epochs_no_improve = 0

        start_time = time.time()

        with tqdm(total=num_epochs, desc="Training Progress") as pbar:
            for epoch in range(num_epochs):
                self.optimizer.zero_grad()

                # Move data generation inside the loop
                x_f, y_f, z_f, t_f, x_bx, y_bx, z_bx, t_bx, x_by, y_by, z_by, t_by, x_bz, y_bz, z_bz, t_bz = self.generate_training_data()

                loss = self.loss_function(x_f, y_f, z_f, t_f,
                                          x_bx, y_bx, z_bx, t_bx,
                                          x_by, y_by, z_by, t_by,
                                          x_bz, y_bz, z_bz, t_bz)

                loss.backward()
                self.optimizer.step()

                current_loss = loss.item()

                # Adaptive loss weighting
                if epoch % adjust_interval == 0 and epoch != 0:
                    # Compute average losses over the interval
                    avg_pde_loss = np.mean(self.pde_loss_history[-adjust_interval:])
                    avg_bc_loss = np.mean(self.bc_loss_history[-adjust_interval:])

                    # Adjust weights inversely proportional to losses
                    total_loss = avg_pde_loss + avg_bc_loss + 1e-8  # Add small epsilon to avoid division by zero
                    self.w_pde = total_loss / (2 * avg_pde_loss + 1e-8)
                    self.w_bc = total_loss / (2 * avg_bc_loss + 1e-8)

                    # Normalize weights to sum to 1
                    weight_sum = self.w_pde + self.w_bc
                    self.w_pde /= weight_sum
                    self.w_bc /= weight_sum

                # Early stopping check
                if best_loss - current_loss > min_delta:
                    best_loss = current_loss
                    epochs_no_improve = 0
                    # Optionally, save the model
                    # torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

                # Update tqdm progress bar
                pbar.set_postfix({
                    'Epoch': epoch,
                    'Loss': f'{loss.item():.5e}',
                    'w_pde': f'{self.w_pde:.3f}',
                    'w_bc': f'{self.w_bc:.3f}',
                    'Elapsed Time': f'{time.time() - start_time:.2f}s'
                })
                pbar.update(1)

    def predict(self, x, y, z, t):
        x = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
        z = torch.tensor(z, dtype=torch.float32).view(-1, 1).to(device)
        t = torch.tensor(t, dtype=torch.float32).view(-1, 1).to(device)

        with torch.no_grad():
            u_pred = self.reparameterize_solution(x, y, z, t)
        return u_pred.cpu().numpy()

    def plot_results(self, t_values, z_slices):
        """
        Plots the solution u(x, y, z, t) at specified time values and z slices.

        Parameters:
        - t_values: List of time values at which to plot the solution.
        - z_slices: List of z values at which to take slices for plotting.
        """
        x_plot = np.linspace(self.domain_bounds['x'][0], self.domain_bounds['x'][1], 50)
        y_plot = np.linspace(self.domain_bounds['y'][0], self.domain_bounds['y'][1], 50)
        X, Y = np.meshgrid(x_plot, y_plot)

        for t_val in t_values:
            for z_val in z_slices:
                # Flatten the grid arrays
                X_flat = X.flatten()
                Y_flat = Y.flatten()
                Z_flat = z_val * np.ones_like(X_flat)
                T_flat = t_val * np.ones_like(X_flat)

                # Predict the solution
                u_pred = self.predict(X_flat, Y_flat, Z_flat, T_flat)
                U = u_pred.reshape(X.shape)

                # Plot the solution u(x, y, z_slice, t_val)
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, U, cmap='viridis')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('u(x, y, z, t)')
                ax.set_title(f't = {t_val}, z = {z_val}')
                plt.show()

    def plot_loss_history(self):
        plt.figure()
        plt.plot(self.total_loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Training Loss History')
        plt.show()
