import torch
import torch.nn as nn
import csv
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.autograd as autograd
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision4ad.datasets import MVTecAD
import torchvision
import math
import wcy

n = 9
V = 50
total_train = 209;
step = 4
alpha =  1

num_big_broken = 20
num_small_broken = 22
num_contamination = 21
num_good = 20

def Get_PCA_for_Thres(opt):


	out = torch.zeros([V*n*n,1000])
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
	idx = 0
	for i,data in enumerate(test_loader,0):
		if (i % step != 0):
			continue
		if (i > step * (V-1)):
			continue
		img, test_label = data
		                            # a 224*224 image can be divided into 9*9 128*128 patches with stride = 12
		a = torch.zeros([n*n,3,128,128])
		for j in range(n):
			for k in range(n):
				a[j*n+k,:,:,:] = img[0,:,j*12:j*12+128,k*12:k*12+128]

		with torch.no_grad():
			output = res(a)
			out[idx*n*n:(idx+1)*n*n,:] = output
			print(idx) 
		idx+=1 

	pca = wcy.RPCA(patchsize,out)
	pca = wcy.standardize(pca)
	print(pca.shape)

	np.savez("pca_v",pca_v = pca );

def Compute_Threshold(opt):

	Get_PCA_for_Thres(opt)
	
	m = 4

	pca_ori = np.load("pca_v.npz")
	pca = pca_ori["pca_v"]
	dictionary = np.load('dic.npz')
	dict = dictionary["arr_0"]
	print(dict.shape)
	print(dict)

	clusters = dict.shape[0]
	distance = np.zeros(pca.shape[0]);
	for i in range(pca.shape[0]):
		dist_sum = 0.0
		p1 = pca[i,:]
		local_dist = np.zeros(clusters)
		for j in range(clusters):
			p2 = dict[j,:]
			local_dist[j] = np.linalg.norm(p1-p2)
		local_dist.sort()

		for j in range(m):
			dist_sum +=  local_dist[j]
		distance[i] = dist_sum / m
	print(distance)
	print(distance.shape)
	dist_avg = np.mean(distance);
	dist_std = np.std(distance);
	threshold = dist_avg + alpha * dist_std
	print(dist_avg)
	print(dist_std)
	print(threshold)
	
	s = 0.0;
	for i in range(pca.shape[0]):
		if (distance[i] > threshold):
			s += 1.0
	anomal_rate = s / pca.shape[0] * 0.8
	print(s / pca.shape[0])
	np.savez("thres",thres = threshold,ad_rate = anomal_rate )
	#dictionary = wcy.clustering(10,pca)
	#print(dictionary)
	#print(dictionary.shape())
	#dictionary = dictionary.tostring()
	#with open("dic.txt","wb+") as f:
	#	f.write(dictionary)


def Generate_PCA_test(opt):
	n = 9

	out = torch.zeros([n*n,1000])
	batchsize = 1
	patchsize = 128
	res = torchvision.models.resnet50(pretrained=True, progress=True)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	transform_test= transforms.Compose([transforms.Resize((256,256)),
    	                            transforms.RandomCrop((224,224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	mvtec_ad = MVTecAD("./dataset", opt.dataset_name, train=False, transform=transform_test)
	test_loader = DataLoader(mvtec_ad, shuffle=False, batch_size=batchsize)
	for i,data in enumerate(test_loader,0):
		img, test_label = data

		                            # a 224*224 image can be divided into 9*9 128*128 patches with stride = 12
		a = torch.zeros([n*n,3,128,128])
		for j in range(n):
			for k in range(n):
				a[j*n+k,:,:,:] = img[0,:,j*12:j*12+128,k*12:k*12+128]

		with torch.no_grad():
			output = res(a)
			if (i == 0):
				out[i*n*n:(i+1)*n*n,:] = output
			else:
				out = torch.vstack((out,output))
			print(i) 

	pca = wcy.RPCA(patchsize,out)
	pca = wcy.standardize(pca)
	print(pca.shape)
	np.savez("pca_test",pca_t = pca )

def compute_accuracy(img_result):
	detect_big_broken = np.sum(img_result[0:num_big_broken])
	detect_small_broken = np.sum(img_result[num_big_broken:num_big_broken + num_small_broken])
	detect_contamination = np.sum(img_result[num_big_broken + num_small_broken:num_big_broken + num_small_broken+num_contamination])
	detect_good_broken = np.sum(img_result[num_big_broken + num_small_broken + num_contamination:num_big_broken + num_small_broken+num_contamination+num_good])
	print("————————————————————————————————————")
	print("Broken: acuracy : ",(detect_big_broken+ detect_small_broken+ detect_contamination )/(num_big_broken+num_small_broken+num_contamination))
	print("Large broken: accuracy : ",detect_big_broken/num_big_broken)
	print("Small broken: accuracy : ",detect_small_broken/num_small_broken)
	print("Contamination: accuracy : ",detect_contamination/num_contamination)
	print("————————————————————————————————————")
	print("Good: accuracy : ",1-detect_good_broken/num_good)
	print("————————————————————————————————————")
def cal(predict,actual):
	#good 0 bad 1
	TP = 0
	FP = 0
	FN = 0
	TN = 0
	for i in range(len(predict)):
		if (predict[i] == 0) and (actual[i] == 0):
			TP += 1
		elif (actual[i] == 1) and (predict[i] == 0):
			FP += 1
		elif (actual[i] == 0) and (predict[i] == 1):	
			FN += 1
		elif (actual[i] == 1) and (predict[i] == 1):
			TN += 1
	print(TP)
	print(FP)
	print(FN)
	print(TN)
	print('sensitivity = ', TP*1.0/(TP+FN))
	print('precision = ',TP*1.0/(TP+FP))
	recall = TP*1.0/(TP+FN)
	precision = TP*1.0/(TP+FP)
	try:
		print('specificity = ',TN*1.0/(TN+FP))
	except:
		print('specificity = 0')
	print('F1-score',2/(1/recall+1/precision))



def test_AD(opt):

	m=4

	Generate_PCA_test(opt)

	pca_ori = np.load("pca_test.npz")
	pca = pca_ori["pca_t"]
	dictionary = np.load('dic.npz')
	dict = dictionary["arr_0"]
	thres_ori = np.load("thres.npz")
	threshold = thres_ori["thres"]
	ad_rate = thres_ori["ad_rate"]
	print(dict.shape)
	print(dict)
	print(threshold)

	clusters = dict.shape[0]
	AD_result = np.zeros(pca.shape[0]);

	for i in range(pca.shape[0]):
		dist_sum = 0.0
		p1 = pca[i,:]
		local_dist = np.zeros(clusters)
		for j in range(clusters):
			p2 = dict[j,:]
			local_dist[j] = np.linalg.norm(p1-p2)
		local_dist.sort()
		for j in range(m):
			dist_sum +=  local_dist[j]
		dist_avg = dist_sum / m
		if (dist_avg > threshold):
			AD_result[i] = 1;
	print(AD_result.shape)
	num_rows = AD_result.shape[0] // (n*n)
	AD_res = AD_result.reshape(num_rows,n*n)
	print(AD_res.shape)

	np.savez("AD_result",res_t = AD_res)
	img_result = np.zeros(AD_res.shape[0])
	rate = []
	for i in range(AD_res.shape[0]):
		num_anomal = np.sum(AD_res[i,:])
		anomal_rate = num_anomal / (n*n)
		rate.append(anomal_rate)
		if (anomal_rate > ad_rate):
			img_result[i] = 1
			print(i," : broken")
		else:
			img_result[i] = 0
			print(i," : good")
	compute_accuracy(img_result)
	mvtec_ad = MVTecAD("./dataset", opt.dataset_name, train=False, transform=transforms.ToTensor())
	test_loader = DataLoader(mvtec_ad, shuffle=False, batch_size=1)
	img_true_label = []
	for i,data in enumerate(test_loader,0):
		img, test_label = data	
		#print(test_label)
		if test_label[0] == 0:
			img_true_label.append(0)
		else:
			img_true_label.append(1)
	cal(img_result,img_true_label)
	with open("score_{}.csv".format(opt.dataset_name), "a") as f:
		f_csv = csv.writer(f)
		f_csv.writerow(['label','rate'])
		for i in range(len(rate)):
			f_csv.writerow([img_true_label[i],rate[i]])

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

	Compute_Threshold(opt)  
	test_AD(opt)
