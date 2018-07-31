# ADMA Demo 2018
The repository contains software library for Data Augmentation Services 

# Requirements
python 3X  
numpy > 1.13  
scipy > 0.19  
pillow > 5.2  
tensorflow (gpu) > 1.3  
MATLAB  

# Stage 1
Contains two parts (i) Training GAN and (ii) Training Ensemble Classifier

## Train GAN
We provide modified version of DCGAN taken from https://github.com/carpedm20/DCGAN-tensorflow
The discriminator and the generator are conditioned on the labels of the images and have less layers.

To execute the GAN and generate images type
python main.py 

This will download CIFAR-10 dataset automatically to the path specified.
The GAN will be trained for category 0 against all with the default parameters specified in the file main.py.
Synthetically Generated Images and associated Labels will be saved in ./Geneated_Data folder.
Example files already exists, ./Generated_Data/Images_10_1_2.mat and ./Generated_Data/Labels_10_0_1.mat

Naming Convention Images_alpha_CategoryA_CategoryB and similarly Labels_alpha_CategoryA_CategoryB

If you wish to change the categories on which GAN is trained then please edit file DCGAN_Modified.py
** Line 162 ** fixed_label = 0 and
** Line 163 ** iter_labels = np.arange((fixed_label + 1), 10)

If you want to specify the split ratio of training dataset while training GAN then edit
** Line 159 **  alpha = 0.1


*** Command line arguments ***** coming soon

## Train Ensemble Classifieris
We train SVM, k-NN and naive Bayes available at MATLAB-R2018a
The codes are in the folder Train_EnsemClass

execute TrainEnsemble.m 

Trained parameters of the classifier's will get save in a mat file "MODEL_X_Y.mat"
X : Category 1
Y : Category 2

*** Arguments description coming soon *****

# Stage 2
Once training of GAN and ensemble classifier is finished move to Stage 2 for filtering synthetic Images and obtaining performance measuer of CNN trained on augmented datasets



