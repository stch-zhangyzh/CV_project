# CV project

This is the final project for CS-172 Computer Version class.

The aim of this project is to apply anomaly detection in [MVTev_AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

Among current SOTA algorithms involves GAN, CAEs, CNN feature dictionary and variation model have been suggested to tackle the problem.

In this project, we use GAN and CNN based model to assess their performance for segmentation and classification of images in [MVTev_AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

Our team members:

Chenyu Wang (wangchy3@shanghaitech.edu.cn)

Yunhao Hu (huyh@shanghaitech.edu.cn)

YiZhou Zhang (zhangyzh@shanghaitech.edu.cn)

Yifeng Li (liyf1@shanghaitech.edu.cn)

Zheng Shu (shuzheng@shanghaitech.edu.cn)

## Installation
`git clone https://github.com/zhangyzh-stch/CV_project`

`cd CV_project/`

`pip install -r requires.txt`

## Run CNN feature

Go to run the CNN feature:

`cd CNN_feature_dictioinary/`

#### Step1: Build the CNN feature dictionary:

`python cnn_feature2.py [dataset_name]`

The `dataset_name` can be select from the MVTev_AD dataset:

bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

We generat the dictionary vectors in `./dic.npz`.

#### Step2: Compute the threashold and the acceptable rate:

`python detection.py [dataset_name]`

We generate the threshold for patch and acceptable rate for image in `./thres.npz`

We generate the list of detection result of each patch in `./AD_result.npz`

## Run AnoGAN

Go to run the AnoGAN:

`cd AnoGAN/`

#### Step1: Train the Generative Adversarial Networks:

`python step1_train_gan.py [dataset_name]`

The `dataset_name` can be select from the MVTev_AD dataset:

bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

(The program will help you download `dataset_name` data in [MVTev_AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/) if you don't have this dataset)

The detailed process is shown in this picture from this paper [f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks](https://www.sciencedirect.com/science/article/pii/S1361841518302640/pdfft?md5=298d7dd0f4f19af16e2548acad5d381a&pid=1-s2.0-S1361841518302640-main.pdf)

![1](https://github.com/zhangyzh-stch/CV_project/blob/main/Pictures/1.png)

You can find the output of generator in `{dataset_name}_results/images/` to check the training processes of the generator.

The trained generator and discriminator will be saved in `{dataset_name}_results/generator` and `{dataset_name}_results/discriminator`

#### Step2: Train the encoder:

`python step2_train_encoder.py [dataset_name]`

The detailed process is shown in this picture from this paper [f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks](https://www.sciencedirect.com/science/article/pii/S1361841518302640/pdfft?md5=298d7dd0f4f19af16e2548acad5d381a&pid=1-s2.0-S1361841518302640-main.pdf)

![2](https://github.com/zhangyzh-stch/CV_project/blob/main/Pictures/2.png)

You can find the output of the reconfigurated pictures in `{dataset_name}_results/images_e/` to check the training processes of the auto-encoder.

The trained encoder will be saved in `{dataset_name}_results/encoder`.

#### Step3: Test anomaly detection:

`python step3_anomaly_detection.py [dataset_name]`

The test result will be saved in `{dataset_name}_results/score.csv` in `label, img_distance, anomaly_score, z_distance` format.

#### Step4: Save the compared images:

`python step4_save_img.py [dataset_name]`

The compared images will be saved in `{dataset_name}_results/images_diff/` with format (real_img, fake_img, real_img - fake_img).

#### Step5: Visualization:

Start the jupyter notebook:

`jupyter notebook`

Open the `AnoGAN.ipynb` file, modify the path to the file `{dataset_name}_results/score.csv`. And then simply run the hole file.

You can get the roc-auc curve and PR curve.

### Reference

The dataset and paper:

[MVTev_AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

[MVTev_AD paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf)

[f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks](https://www.sciencedirect.com/science/article/pii/S1361841518302640/pdfft?md5=298d7dd0f4f19af16e2548acad5d381a&pid=1-s2.0-S1361841518302640-main.pdf)

The reference code of this project is here:

https://github.com/LeeDoYup/AnoGAN-tf

https://github.com/eriklindernoren/PyTorch-GAN

https://github.com/A03ki/f-AnoGAN
