import io
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

def make_grid(X, spaces = 100):
  device = X.device
  X = X.detach().cpu().numpy()
  min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
  min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1

  x1grid = np.arange(min1, max1, (max1 - min1) / spaces)
  x2grid = np.arange(min2, max2, (max2 - min2) / spaces)

  xx, yy = np.meshgrid(x1grid, x2grid)

  r1, r2 = xx.flatten(), yy.flatten()
  r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

  grid = np.hstack((r1,r2))

  return (xx, yy), torch.tensor(grid).float().to(device)