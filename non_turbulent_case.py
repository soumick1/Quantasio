import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from matplotlib.animation import FuncAnimation, PillowWriter
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
            out = out + residual  
        out = self.output_layer(out)
        return out

class Solver4D:
    def __init__(self, pde_residual, initial_condition_function, domain_bounds, model_params, training_params):
        self.pde_residual = pde_residual
        self.initial_condition_function = initial_condition_function
        self.domain_bounds = domain_bounds
        self.model_params = model_params
        self.training_params = training_params

        self.model = ResNet4D(
            num_inputs=4,
            num_outputs=4,  # u, v, w, p
            num_hidden_layers=model_params.get('num_hidden_layers', 4),
            num_neurons=model_params.get('num_neurons', 50)
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=training_params.get('learning_rate', 1e-3))

        self.pde_loss_history = []
        self.div_loss_history = []
        self.total_loss_history = []

    def reparameterize_solution(self, x, y, z, t):
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        z = z.view(-1, 1)
        t = t.view(-1, 1)

        # Neural network output (u, v, w, p)
        N = self.model(x, y, z, t)

        # Initial conditions
        u0, v0, w0, p0 = self.initial_condition_function(x, y, z)

        u = u0 + t * N[:, 0:1]
        v = v0 + t * N[:, 1:2]
        w = w0 + t * N[:, 2:3]
        p = p0 + t * N[:, 3:4]
        return u, v, w, p

    def generate_training_data(self):
        N_f = self.training_params.get('N_f', 10000)
        x_f = torch.rand(N_f, 1, requires_grad=True).to(device)
        y_f = torch.rand(N_f, 1, requires_grad=True).to(device)
        z_f = torch.rand(N_f, 1, requires_grad=True).to(device)
        t_f = torch.rand(N_f, 1, requires_grad=True).to(device)

        return x_f, y_f, z_f, t_f

    def loss_function(self, x_f, y_f, z_f, t_f):
        f_u, f_v, f_w, div = self.pde_residual(x_f, y_f, z_f, t_f, self.reparameterize_solution)
        mse_pde = torch.mean(f_u ** 2 + f_v ** 2 + f_w ** 2)
        mse_div = torch.mean(div ** 2)

        self.pde_loss_history.append(mse_pde.item())
        self.div_loss_history.append(mse_div.item())
        self.total_loss_history.append((mse_pde + mse_div).item())

        return mse_pde + mse_div

    def train(self):
        num_epochs = self.training_params.get('num_epochs', 5000)
        patience = self.training_params.get('patience', 500)
        min_delta = self.training_params.get('min_delta', 1e-6)

        best_loss = np.inf
        epochs_no_improve = 0

        with tqdm(total=num_epochs, desc="Training Progress") as pbar:
            for epoch in range(num_epochs):
                self.optimizer.zero_grad()
                x_f, y_f, z_f, t_f = self.generate_training_data()
                loss = self.loss_function(x_f, y_f, z_f, t_f)
                loss.backward()
                self.optimizer.step()

                current_loss = loss.item()
                if best_loss - current_loss > min_delta:
                    best_loss = current_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

                pbar.set_postfix({'Epoch': epoch, 'Loss': f'{loss.item():.5e}'})
                pbar.update(1)

    def predict(self, x, y, z, t):
        x = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
        z = torch.tensor(z, dtype=torch.float32).view(-1, 1).to(device)
        t = torch.tensor(t, dtype=torch.float32).view(-1, 1).to(device)

        with torch.no_grad():
            u, v, w, p = self.reparameterize_solution(x, y, z, t)
        return u.cpu().numpy(), v.cpu().numpy(), w.cpu().numpy(), p.cpu().numpy()

    def plot_loss_history(self):
        plt.figure()
        plt.plot(self.pde_loss_history, label="PDE Loss")
        plt.plot(self.div_loss_history, label="Boundary Loss")
        plt.plot(self.total_loss_history, label="Total Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss History')
        plt.show()

    def animate_solution(self, time_steps, resolution=20, save_path=None, save_format="mp4"):
        """
        Animate the Navier-Stokes solution over time.

        Parameters:
        - time_steps: List of time steps for the animation.
        - resolution: Number of points in each spatial dimension for the grid.
        - save_path: File path to save the animation. If None, the animation is only shown.
        - save_format: Format to save the animation ('mp4' or 'gif').
        """
        x = np.linspace(self.domain_bounds['x'][0], self.domain_bounds['x'][1], resolution)
        y = np.linspace(self.domain_bounds['y'][0], self.domain_bounds['y'][1], resolution)
        z = np.linspace(self.domain_bounds['z'][0], self.domain_bounds['z'][1], resolution)
        X, Y, Z = np.meshgrid(x, y, z)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            ax.clear()
            T = np.full_like(X, time_steps[frame])
            u, v, w, p = self.predict(X.flatten(), Y.flatten(), Z.flatten(), T.flatten())
            u, v, w = u.reshape(X.shape), v.reshape(Y.shape), w.reshape(Z.shape)
            ax.quiver(X, Y, Z, u, v, w, length=0.1, normalize=True)
            ax.set_title(f"Velocity Field at t = {time_steps[frame]:.2f}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        ani = FuncAnimation(fig, update, frames=len(time_steps), interval=200)

        if save_path:
            try:
                if save_format == "gif":
                    ani.save(save_path, writer=PillowWriter(fps=10))
                elif save_format == "mp4":
                    ani.save(save_path, writer="ffmpeg")
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Error saving animation: {e}. Ensure ffmpeg or Pillow is installed.")

        plt.show()

    def animate_solution_vector_field(self, time_steps, resolution=20, save_path=None, save_format="mp4"):
        """
        Animate the Navier-Stokes solution over time, showing the vector field (u, v, w).

        Parameters:
        - time_steps: List of time steps for the animation.
        - resolution: Number of points in each spatial dimension for the grid.
        - save_path: File path to save the animation. If None, the animation is only shown.
        - save_format: Format to save the animation ('mp4' or 'gif').
        """
        x = np.linspace(self.domain_bounds['x'][0], self.domain_bounds['x'][1], resolution)
        y = np.linspace(self.domain_bounds['y'][0], self.domain_bounds['y'][1], resolution)
        z = np.linspace(self.domain_bounds['z'][0], self.domain_bounds['z'][1], resolution)
        X, Y, Z = np.meshgrid(x, y, z)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            ax.clear()
            T = np.full_like(X, time_steps[frame])
            u, v, w, p = self.predict(X.flatten(), Y.flatten(), Z.flatten(), T.flatten())
            u, v, w = u.reshape(X.shape), v.reshape(Y.shape), w.reshape(Z.shape)

            ax.quiver(
                X[::2, ::2, ::2], Y[::2, ::2, ::2], Z[::2, ::2, ::2],
                u[::2, ::2, ::2], v[::2, ::2, ::2], w[::2, ::2, ::2],
                length=0.1, normalize=True, color="blue"
            )

            ax.set_title(f"Velocity Field at t = {time_steps[frame]:.2f}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        ani = FuncAnimation(fig, update, frames=len(time_steps), interval=200)

        if save_path:
            try:
                if save_format == "mp4":
                    ani.save(save_path, writer="ffmpeg")
                elif save_format == "gif":
                    ani.save(save_path, writer=PillowWriter(fps=10))
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Error saving animation: {e}. Try installing ffmpeg or using GIF format.")

        plt.show()

    def plot_solution_vector_field_at_time(self, time_instance, resolution=20, save_path=None):
        """
        Plot the Navier-Stokes solution vector field at a specific time instance.

        Parameters:
        - time_instance: The specific time at which to plot the solution.
        - resolution: Number of points in each spatial dimension for the grid.
        """
        x = np.linspace(self.domain_bounds['x'][0], self.domain_bounds['x'][1], resolution)
        y = np.linspace(self.domain_bounds['y'][0], self.domain_bounds['y'][1], resolution)
        z = np.linspace(self.domain_bounds['z'][0], self.domain_bounds['z'][1], resolution)
        X, Y, Z = np.meshgrid(x, y, z)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        T = np.full_like(X, time_instance)
        u, v, w, p = self.predict(X.flatten(), Y.flatten(), Z.flatten(), T.flatten())
        u, v, w = u.reshape(X.shape), v.reshape(Y.shape), w.reshape(Z.shape)

        ax.quiver(
            X[::2, ::2, ::2], Y[::2, ::2, ::2], Z[::2, ::2, ::2],
            u[::2, ::2, ::2], v[::2, ::2, ::2], w[::2, ::2, ::2],
            length=0.1, normalize=True, color="blue"
        )

        ax.set_title(f"Velocity Field at time t = {time_instance:.2f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot for vector field at time {time_instance} saved to {save_path}")

        plt.show()

    def animate_pressure_field(self, time_steps, resolution=20, save_path=None, save_format="mp4"):
        """
        Animate the Navier-Stokes pressure field over time.

        Parameters:
        - time_steps: List of time steps for the animation.
        - resolution: Number of points in each spatial dimension for the grid.
        - save_path: File path to save the animation. If None, the animation is only shown.
        - save_format: Format to save the animation ('mp4' or 'gif').
        """
        x = np.linspace(self.domain_bounds['x'][0], self.domain_bounds['x'][1], resolution)
        y = np.linspace(self.domain_bounds['y'][0], self.domain_bounds['y'][1], resolution)
        z = np.linspace(self.domain_bounds['z'][0], self.domain_bounds['z'][1], resolution)
        X, Y, Z = np.meshgrid(x, y, z)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        cbar = None  

        def update(frame):
            nonlocal cbar
            ax.clear()
            if cbar:
                cbar.remove()  
                cbar = None

            T = np.full_like(X[:, :, 0], time_steps[frame])
            try:
                _, _, _, p = self.predict(X.flatten(), Y.flatten(), Z.flatten(), T.flatten())
            except Exception as e:
                print(f"Error during prediction: {e}")
                return

            p = p.reshape(X.shape)

           
            surf = ax.plot_surface(
                X[:, :, 0], Y[:, :, 0], p[:, :, 0], cmap='viridis', edgecolor='none', alpha=0.8
            )
            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
            cbar.set_label('Pressure')

            ax.set_title(f"Pressure Field at t = {time_steps[frame]:.2f}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Pressure")

        ani = FuncAnimation(fig, update, frames=len(time_steps), interval=200)

        if save_path:
            try:
                if save_format == "mp4":
                    ani.save(save_path, writer="ffmpeg")
                elif save_format == "gif":
                    ani.save(save_path, writer=PillowWriter(fps=10))
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Error saving animation: {e}. Try installing ffmpeg or using GIF format.")

        plt.show()

    def visualize_pressure_map(self, time_instance, resolution=20, save_path=None):
        """
        Visualize the Navier-Stokes pressure field at a specific timestamp.

        Parameters:
        - time_instance: The specific time at which to visualize the pressure field.
        - resolution: Number of points in each spatial dimension for the grid.
        - save_path: File path to save the visualization. If None, the visualization is only shown.
        """
        
        x = np.linspace(self.domain_bounds['x'][0], self.domain_bounds['x'][1], resolution)
        y = np.linspace(self.domain_bounds['y'][0], self.domain_bounds['y'][1], resolution)
        z = np.linspace(self.domain_bounds['z'][0], self.domain_bounds['z'][1], resolution)
        X, Y, Z = np.meshgrid(x, y, z)

        
        T = np.full_like(X.flatten(), time_instance)

        
        try:
            _, _, _, p = self.predict(X.flatten(), Y.flatten(), Z.flatten(), T)
            p = p.reshape(X.shape)  
        except Exception as e:
            print(f"Error during prediction: {e}")
            return

        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        
        surf = ax.plot_surface(
            X[:, :, 0], Y[:, :, 0], p[:, :, 0], cmap='viridis', edgecolor='none', alpha=0.8
        )
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Pressure')

        
        ax.set_title(f"Pressure Field at t = {time_instance:.2f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Pressure")

        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Pressure field visualization saved to {save_path}")

        
        plt.show()


def navier_stokes_residual(x, y, z, t, solution_fn):
    u, v, w, p = solution_fn(x, y, z, t)

    u_t = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]

    v_t = grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_z = grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_zz = grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]

    w_t = grad(w, t, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_x = grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_y = grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_z = grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_xx = grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_yy = grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
    w_zz = grad(w_z, z, grad_outputs=torch.ones_like(w_z), create_graph=True)[0]

    p_x = grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_z = grad(p, z, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    nu = 0.1  
    f_u = u_t + u * u_x + v * u_y + w * u_z + p_x - nu * (u_xx + u_yy + u_zz)
    f_v = v_t + u * v_x + v * v_y + w * v_z + p_y - nu * (v_xx + v_yy + v_zz)
    f_w = w_t + u * w_x + v * w_y + w * w_z + p_z - nu * (w_xx + w_yy + w_zz)

    div = u_x + v_y + w_z
    return f_u, f_v, f_w, div

def initial_conditions(x, y, z):
    u0 = torch.zeros_like(x)
    v0 = torch.zeros_like(y)
    w0 = torch.zeros_like(z)
    p0 = torch.zeros_like(x)
    return u0, v0, w0, p0


domain_bounds = {
    'x': (0, 1),
    'y': (0, 1),
    'z': (0, 1),
    't': (0, 1)
}


model_params = {
    'num_hidden_layers': 6,
    'num_neurons': 100
}


training_params = {
    'num_epochs': 5000,
    'learning_rate': 1e-3,
    'N_f': 10000,
    'patience': 100,
    'min_delta': 1e-6
}


solver = Solver4D(
    pde_residual=navier_stokes_residual,
    initial_condition_function=initial_conditions,
    domain_bounds=domain_bounds,
    model_params=model_params,
    training_params=training_params
)


solver.train()


solver.plot_loss_history()


time_steps = np.linspace(0, 1, 600)
solver.animate_solution(
    time_steps=time_steps,
    resolution=10,
    save_path="navier_stokes_animation_np.gif",
    save_format="gif"
)


solver.visualize_pressure_map(
    time_instance=0.0,
    resolution=10,
    save_path="pressure_map_np_t0.png"
)

solver.visualize_pressure_map(
    time_instance=0.5,
    resolution=10,
    save_path="pressure_map_np_t0.5.png"
)

solver.visualize_pressure_map(
    time_instance=1.0,
    resolution=10,
    save_path="pressure_map_np_t1.png"
)
