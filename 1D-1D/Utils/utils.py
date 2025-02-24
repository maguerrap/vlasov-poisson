import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.size': 14,  # General font size
    'axes.labelsize': 14,  # Axis label size
    'axes.titlesize': 14,  # Title size
    'xtick.labelsize': 14,  # X-tick label size
    'ytick.labelsize': 14,  # Y-tick label size
    'legend.fontsize': 14   # Legend font size
})


### Plotting functions
# Function to plot the distribution of f_eq over x and v
def plot_feq_distribution(fig, ax, f_eq, title, mesh):
    im = ax.imshow(f_eq.transpose(), extent=[mesh.xs[0], mesh.xs[-1], mesh.vs[0], mesh.vs[-1]], aspect='auto', cmap='plasma')  # 'auto' aspect ratio
    ax.set_title(title)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$v$')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Adjusted colorbar size

# Modify the plot_distribution function to handle 2D data
def plot_distribution(fig, ax, data, title, time, mesh):
    im = ax.imshow(data.T, extent=[mesh.xs[0], mesh.xs[-1], mesh.vs[0], mesh.vs[-1]], aspect='auto', cmap='plasma')  # Transpose data
    ax.set_title(f'{title} (T={time:.2f})')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$v$')
    fig.colorbar(im, ax=ax)#, fraction=0.046, pad=0.04)  # Adjusted colorbar size