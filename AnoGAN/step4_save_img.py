import os
import torch

from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision4ad.datasets import MVTecAD

from model import Generator, Encoder

"""
The codes are modified based on:
Copyright (c) 2020 Aki
Licensed under MIT
(https://github.com/A03ki/f-AnoGAN/blob/master/LICENSE)
And it has been changed for our real usage. 
"""


def save_img(opt, generator, encoder, dataloader, device):
    generator.load_state_dict(torch.load("{}_results/generator".format(opt.dataset_name)))
    encoder.load_state_dict(torch.load("{}_results/encoder".format(opt.dataset_name)))

    generator.to(device).eval()
    encoder.to(device).eval()

    os.makedirs("{}_results/images_diff".format(opt.dataset_name), exist_ok=True)

    for i, (img, label) in enumerate(dataloader):
        real_img = img.to(device)

        real_z = encoder(real_img)
        fake_img = generator(real_z)

        compared_images = torch.empty(real_img.shape[0] * 3,
                                      *real_img.shape[1:])
        compared_images[0::3] = real_img
        compared_images[1::3] = fake_img
        compared_images[2::3] = real_img - fake_img

        save_image(compared_images.data,
                   f"{opt.dataset_name}_results/images_diff/{opt.n_grid_lines*(i+1):06}.png",
                   nrow=3, normalize=True)

        if opt.n_iters is not None and opt.n_iters == i:
            break


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize([opt.img_size]*2),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5],
                                                         [0.5, 0.5, 0.5])])
    mvtec_ad = MVTecAD(".", opt.dataset_name, train=False, transform=transform,
                       download=True)
    test_dataloader = DataLoader(mvtec_ad, batch_size=opt.n_grid_lines,
                                 shuffle=False)

    generator = Generator(opt)
    encoder = Encoder(opt)

    save_img(opt, generator, encoder, test_dataloader, device)


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_grid_lines", type=int, default=1,
                        help="number of grid lines in the saved image")
    parser.add_argument("dataset_name", type=str,
                        choices=MVTecAD.available_dataset_names,
                        help="name of MVTec Anomaly Detection Datasets")
    parser.add_argument("--force_download", "-f", action="store_true",
                        help="flag of force download")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels")
    parser.add_argument("--n_iters", type=int, default=None,
                        help="value of stopping iterations")
    opt = parser.parse_args()

    main(opt)
