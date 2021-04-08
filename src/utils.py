import numpy as np
import torch
import torch.nn.functional as F

celeba_cmap = np.array(
    [(0, 0, 0), (204, 0, 0), (76, 153, 0),
    (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
    (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
    (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
    (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
    dtype=np.uint8
)

maadaa_cmap = np.array(
    [(0, 0, 0), (204, 0, 0), (76, 153, 0),
    (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
    (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
    (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
    (0, 204, 204), (0, 51, 0)],
    dtype=np.uint8
)


def colorize_maadaa(gray_image, device="cuda"):
    n_colors = 3
    cmap = torch.from_numpy(maadaa_cmap).to(device)
    shape = gray_image.size()
    color_image = torch.ByteTensor(3, shape[1], shape[2]).fill_(0).to(device)

    for label in range(0, len(cmap)):
        label_mask = (label == gray_image[0]).to(device)
        for i in range(n_colors):
            color_image[i][label_mask] = cmap[label][i]
    return color_image


def colorize_celeba(gray_image, device="cuda"):
    n_colors = 3
    cmap = torch.from_numpy(celeba_cmap).to(device)
    shape = gray_image.size()
    color_image = torch.ByteTensor(3, shape[1], shape[2]).fill_(0).to(device)

    for label in range(0, len(cmap)):
        label_mask = (label == gray_image[0]).to(device)
        for i in range(n_colors):
            color_image[i][label_mask] = cmap[label][i]
    return color_image


def denorm(tensor):
    out = (tensor + 1) / 2
    return out.clamp_(0, 1)


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear",
                              align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def tensor_to_label(label_tensor, imtype=np.uint8, device="cuda", to_numpy=False, dataset="celeba"):
    label_tensor = label_tensor.to(device).float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    if dataset == "celeba":
        label = colorize_celeba(label_tensor)
    elif dataset == "maadaa":
        label = colorize_maadaa(label_tensor)
    if to_numpy:
        label = label_tensor.cpu().numpy()
    label = label / 255.0
    return label


def generate_mask(mask_tensor, image_size):
    masks = []
    for mask in mask_tensor:
        mask = mask.view(1, 19, image_size, image_size)
        mask = mask.data.max(1)[1]
        mask = tensor_to_label(mask)
        masks.append(mask)
    masks = torch.stack(masks)
    return masks
