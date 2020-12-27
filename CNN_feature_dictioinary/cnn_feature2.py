import torch
import torch.nn as nn

import numpy as np
from torchvision4ad.datasets import MVTecAD
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.autograd as autograd
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision4ad.datasets import MVTecAD
import torchvision
import math
import wcy

def main(opt):
	n = 9
	
	batchsize = 1
	patchsize = 128
	res = torchvision.models.resnet50(pretrained=True, progress=True)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	transform_test= transforms.Compose([transforms.Resize((256,256)),
    	                            transforms.RandomCrop((224,224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	mvtec_ad = MVTecAD("./dataset", opt.dataset_name, train=True, transform=transform_test)
	test_loader = DataLoader(mvtec_ad, shuffle=False, batch_size=batchsize)
	c = 0
	for i,data in enumerate(test_loader,0):
		c += 1
	out = torch.zeros([c*n*n,1000])	
	for i,data in enumerate(test_loader,0):
		img, test_label = data

		                            # a 224*224 image can be divided into 9*9 128*128 patches with stride = 12
		a = torch.zeros([n*n,3,128,128])
		for j in range(n):
			for k in range(n):
				a[j*n+k,:,:,:] = img[0,:,j*12:j*12+128,k*12:k*12+128]

		with torch.no_grad():
			output = res(a)
			out[i*n*n:(i+1)*n*n,:] = output
			print(i) 

	pca = wcy.RPCA(patchsize,out)
	pca = wcy.standardize(pca)

	#print(len(pca))
	print(len(pca[0]))
	dictionary = wcy.clustering(100,pca)

	np.savez('dic.npz',dictionary)

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
 
	opt = parser.parse_args()

	main(opt)  