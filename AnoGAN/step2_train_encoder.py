import os
import torch

import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision4ad.datasets import MVTecAD

from model import Generator, Discriminator, Encoder

"""
The codes are modified based on:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
And it has been changed for our real usage. 
"""

def train_encoder(opt, generator, discriminator, encoder,
                       dataloader, device, kappa=1.0):
    generator.load_state_dict(torch.load("{}_results/generator".format(opt.dataset_name)))
    discriminator.load_state_dict(torch.load("{}_results/discriminator".format(opt.dataset_name)))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device)

    criterion = nn.MSELoss()

    optimizer_E = torch.optim.Adam(encoder.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))

    os.makedirs("{}_results/images_e".format(opt.dataset_name), exist_ok=True)

    padding_epoch = len(str(opt.n_epochs))
    padding_i = len(str(len(dataloader)))

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = imgs.to(device)

            # ----------------
            #  Train Encoder
            # ----------------

            optimizer_E.zero_grad()

            # Generate a batch of latent variables
            z = encoder(real_imgs)

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real features
            real_features = discriminator.forward_features(real_imgs)
            # Fake features
            fake_features = discriminator.forward_features(fake_imgs)

            # izif architecture
            loss_imgs = criterion(fake_imgs, real_imgs)
            loss_features = criterion(fake_features, real_features)
            e_loss = loss_imgs + kappa * loss_features

            e_loss.backward()
            optimizer_E.step()

            # Output training log every n_critic steps
            if i % opt.n_critic == 0:
                print(f"[Epoch {epoch:{padding_epoch}}/{opt.n_epochs}] "
                      f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                      f"[E loss: {e_loss.item():3f}]")

                if batches_done % opt.sample_interval == 0:
                    fake_z = encoder(fake_imgs)
                    reconfiguration_imgs = generator(fake_z)
                    save_image(reconfiguration_imgs.data[:25],
                               f"{opt.dataset_name}_results/images_e/{batches_done:06}.png",
                               nrow=5, normalize=True)

                batches_done += opt.n_critic
    torch.save(encoder.state_dict(), "{}_results/encoder".format(opt.dataset_name))

def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize([opt.img_size]*2),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5],
                                                         [0.5, 0.5, 0.5])])
    mvtec_ad = MVTecAD(".", opt.dataset_name, train=True, transform=transform,
                       download=True)
    train_dataloader = DataLoader(mvtec_ad, batch_size=opt.batch_size,
                                  shuffle=True)

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    train_encoder(opt, generator, discriminator, encoder,
                       train_dataloader, device)


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str,
                        choices=MVTecAD.available_dataset_names,
                        help="name of MVTec Anomaly Detection Datasets")
    parser.add_argument("--force_download", "-f", action="store_true",
                        help="flag of force download")
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=400,
                        help="interval betwen image samples")
    parser.add_argument("--seed", type=int, default=None,
                        help="value of a random seed")
    opt = parser.parse_args()

    main(opt)
