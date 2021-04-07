import numpy as np
import torch

celeba_cmap = np.array(
    [(0, 0, 0), (204, 0, 0), (76, 153, 0),
    (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
    (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
    (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
    (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
    dtype=np.uint8
)

def colorize_celeba(gray_image, device="cuda"):
    n_colors = 3
    cmap = torch.from_numpy(celeba_cmap, device=device)
    shape = gray_image.size()
    color_image = torch.ByteTensor(3, shape[1], shape[3]).fill_(0).to(device)

    for label in range(0, len(cmap)):
        label_mask = (label == gray_image[0]).to(device)
        for i in range(n_colors):
            color_image[i][label_mask] = cmap[label][i]
    return color_image

def denorm(tensor):
    tensor = (tensor + 1) / 2
    return tensor.clamp_(0, 1)