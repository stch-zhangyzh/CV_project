import torch

import torch.nn as nn
from torch.utils.model_zoo import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision4ad.datasets import MVTecAD

from model import Generator, Discriminator, Encoder

"""
The codes are modified based on:
Copyright (c) 2020 Aki
Licensed under MIT
(https://github.com/A03ki/f-AnoGAN/blob/master/LICENSE)
And it has been changed for our real usage. 
"""

def anomaly_detection(opt, generator, discriminator, encoder, dataloader, device):
    generator.load_state_dict(torch.load("{}_results/generator".format(opt.dataset_name)))
    discriminator.load_state_dict(torch.load("{}_results/discriminator".format(opt.dataset_name)))
    encoder.load_state_dict(torch.load("{}_results/encoder".format(opt.dataset_name)))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device).eval()

    criterion = nn.MSELoss()

    with open("{}_results/score.csv".format(opt.dataset_name), "w") as f:
        f.write("label, img_distance, anomaly_score, z_distance\n")

    for (img, label) in tqdm(dataloader):
        real_img = img.to(device)
        real_z = encoder(real_img)
        fake_img = generator(real_z)
        fake_z = encoder(fake_img)

        # get f(x) and f(G(x))
        real_feature = discriminator.forward_features(real_img)
        fake_feature = discriminator.forward_features(fake_img)

        # get img_distance, anomaly_score, z_distance
        img_distance = criterion(fake_img, real_img)
        feature_loss = criterion(fake_feature, real_feature)
        anomaly_score = img_distance + feature_loss
        z_distance = criterion(fake_z, real_z)

        with open("{}_results/score.csv".format(opt.dataset_name), "a") as f:
            f.write("{}, {}, {}, {}\n".format(label.item(), img_distance, anomaly_score, z_distance))


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize([opt.img_size]*2),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5],
                                                         [0.5, 0.5, 0.5])])
    mvtec_ad = MVTecAD(".", opt.dataset_name, train=False, transform=transform,
                       download=True)
    test_dataloader = DataLoader(mvtec_ad, batch_size=1, shuffle=False)

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    anomaly_detection(opt, generator, discriminator, encoder, test_dataloader, device)



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
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels")
    opt = parser.parse_args()

    main(opt)
