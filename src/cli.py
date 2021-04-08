from dataloader import CelebAMaskHQLoader, MaadaaLoader
import fire
import time
import datetime
from pathlib import Path
import torch
from torch import nn
from torchvision.utils import save_image
from utils import colorize_celeba, denorm, cross_entropy2d, generate_mask
from model import UNet
from preprocessing import mkdir_if_empty_or_not_exist

def train_from_folder(
    data_dir = "../data",
    results_dir = "../data/results",
    models_dir = "../",
    image_size = 512,
    version = "parsenet",
    total_step = 1000000,
    batch_size = 4,
    accumulation_steps = 2,
    n_workers = 2,
    learning_rate = 0.0002,
    lr_decay = 0.95,
    beta1 = 0.5,
    beta2 = 0.999,
    test_size = 2824,
    model_name = "model.pth",
    pretrained_model = None,
    is_train = True,
    parallel = False,
    use_tensorboard = False,
    image_path = "../data/maadaa/train/image",
    mask_path = "../data/maadaa/train/seg",
    log_path = "./logs",
    model_save_path = "./models",
    sample_path = "./samples",
    test_image_path = "../data/maada/val/image",
    test_mask_path = "./test_results",
    test_color_mask_path = "./test_color_visualize",
    log_step = 10,
    sample_step = 100,
    model_save_step = 1.0,
    device = "cuda",
    verbose = 1,
    dataset = "maadaa"
):
    sample_path = Path(sample_path)
    model_save_path = Path(model_save_path)
    mkdir_if_empty_or_not_exist(sample_path)
    mkdir_if_empty_or_not_exist(model_save_path)

    print(sample_path)

    if dataset == "celeba":
        dataloader = CelebAMaskHQLoader(image_path, mask_path, image_size,
                                        batch_size, is_train).loader()
    elif dataset == "maadaa":
        dataloader = MaadaaLoader(image_path, mask_path, image_size,
                                  batch_size, is_train).loader()

    # data iterator
    data_iter = iter(dataloader)
    step_per_epoch = len(dataloader)
    model_save_step = int(model_save_step * step_per_epoch)

    # start with pretrained model
    if pretrained_model:
        start = pretrained_model + 1
    else:
        start = 0

    network = UNet().to(device)
    if parallel:
        network = nn.DataParallel(network)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()),
                                 learning_rate, [beta1, beta2])

    if verbose > 0:
        print(network)

    # start time
    start_time = time.time()
    for step in range(start, total_step):
        network.train()
        try:
            images, labels = next(data_iter)
        except:
            data_iter = iter(dataloader)
            images, labels = next(data_iter)

        shape = labels.size()

        labels = (labels * 255.0).to(device)
        labels_true_plain = labels.squeeze(1)
        one_hot_size = (shape[0], 19, shape[2], shape[3])
        labels_true = torch.FloatTensor(torch.Size(one_hot_size)).zero_().to(device)
        labels_true = labels_true.scatter_(1, labels.data.long(), 1.0)

        images = images.to(device)

        # train network
        labels_predict = network(images)

        loss = cross_entropy2d(labels_predict, labels_true_plain.long())
        loss.backward()

        if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
            optimizer.step()
            optimizer.zero_grad()


        if verbose > 0:
            if (step + 1) % log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"Elapsed [elapsed], step [{step + 1} / {total_step}], "
                      f"loss: {loss.item():.4f}")

        #masks_pred = generate_mask(labels_predict, image_size)
        #masks_true = generate_mask(labels_true, image_size)

        # add tensorboard logging later

        if (step + 1) % sample_step == 0:
            with torch.no_grad():
                network.eval()
                masks_sample = network(images)
            masks_sample = generate_mask(masks_sample, image_size)
            masks_true = generate_mask(labels_true.data, image_size)
            save_image(denorm(masks_sample.data),
                       str(sample_path / f"{step+1}_predict.png"))
            save_image(images.data,
                       str(sample_path / f"{step + 1}_images.png"))
            save_image(denorm(masks_true.data),
                       str(sample_path / f"{step + 1}_true.png"))

        if (step + 1) % model_save_step == 0:
            torch.save(network.state_dict(),
                       str(model_save_path / f"{step + 1}_network.pth"))


def main():
    #fire.Fire(train_from_folder)
    train_from_folder()

if __name__ == "__main__":
    main()