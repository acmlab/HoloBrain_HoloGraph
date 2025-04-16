import torch
import matplotlib.pyplot as plt

# Set random seed
torch.manual_seed(0)

# Define a 5-node graph
W = torch.tensor([
    [0., 1., 0., 0., 0.],
    [1., 0., 1., 0., 0.],
    [0., 1., 0., 1., 0.],
    [0., 0., 1., 0., 1.],
    [0., 0., 0., 1., 0.]
])

# Initial node states
x_diff = torch.rand(5)
x_osc = x_diff.clone()

# Parameters
timesteps = 80
gamma = 0.4
omega = torch.zeros(5)
trajectory_diff = [x_diff.numpy()]
trajectory_osc = [x_osc.numpy()]

# Nonlinear control function
def phi(x):
    return torch.tanh(x)

# Diffusion dynamics
for _ in range(timesteps):
    x_diff = x_diff + 0.2 * (W @ x_diff - x_diff)
    trajectory_diff.append(x_diff.numpy())

# Oscillator dynamics
for _ in range(timesteps):
    x_osc = x_osc + 0.2 * (omega + gamma * phi(W @ x_osc))
    trajectory_osc.append(x_osc.numpy())

trajectory_diff = torch.tensor(trajectory_diff)
trajectory_osc = torch.tensor(trajectory_osc)

plt.figure(figsize=(12, 5))

# --- Diffusion Dynamics ---
plt.subplot(1, 2, 1)
for i in range(5):
    plt.plot(trajectory_diff[:, i], label=f'Node {i}', linewidth=2)
plt.title("Diffusion Dynamics", fontweight='bold', fontsize=14)
plt.xlabel("Time step", fontweight='bold', fontsize=12)
plt.ylabel("Node state", fontweight='bold', fontsize=12)
plt.legend()

# --- Oscillator Dynamics ---
plt.subplot(1, 2, 2)
for i in range(5):
    plt.plot(trajectory_osc[:, i], label=f'Node {i}', linewidth=2)
plt.title("Oscillator Dynamics", fontweight='bold', fontsize=14)
plt.xlabel("Time step", fontweight='bold', fontsize=12)
plt.ylabel("Node state", fontweight='bold', fontsize=12)
plt.legend()

plt.tight_layout()
plt.savefig("diffusion_vs_oscillator_dynamics_80.svg", dpi=300)
print("Figure saved as: diffusion_vs_oscillator_dynamics.svg")
