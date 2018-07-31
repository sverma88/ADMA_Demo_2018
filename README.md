# ADMA Demo 2018
The repository contains software library for __Data Augmentation Services__

# Requirements
python 3.x  
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
Synthetically Generated Images and associated Labels will be saved in __./Geneated_Data folder__.  
Example files already exists, __./Generated_Data/Images_10_1_2.mat__ and __./Generated_Data/Labels_10_0_1.mat__

Naming Convention Images_alpha_CategoryA_CategoryB and similarly Labels_alpha_CategoryA_CategoryB

If you wish to change the categories on which GAN is trained then please edit file DCGAN_Modified.py  
**Line 162, fixed_label = 0** and
**Line 163, iter_labels = np.arange((fixed_label + 1), 10)**  

If you want to specify the split ratio of training dataset while training GAN then edit  
**Line 159, alpha = 0.1**  

## Train Ensemble Classifieris
We train SVM, k-NN and naive Bayes available at MATLAB-R2018a
The codes are in the folder __Train_EnsemClass__  

To train Ensemble classifier execute __TrainEnsemble.m__   

Trained parameters of the classifier's will get save in a mat file __MODEL_X_Y.mat__, example file exists in the folder   
X : Category 1
Y : Category 2  

You can specify on which labels you want to train your ensemble classifier then edit file __TrainEnsemble.m__  
**Line 14, fixed_label = 1**  
**Line 15, selected_labels = [(fixed_label+1):10]**  


# Stage 2
Once training of then GAN and ensemble classifier is finished and outputs are saved in their corresponding locations. You can move to __Stage-2__ for filtering synthetic Images and obtaining performance measuer of CNN trained on augmented datasets.  

## Filter Unbiased Images
Execute Filter_Images.m file to filter the synthetic images by trained ensemble classifier.  

Path of the training data, the saved model, and the generated data is required. They are already set, but if you change the save path of any file then please modify them as below:  
__Line 16, path of the trained ensemble classifier's model__  
__Line 17, path of the training data__  
__Line 78, path of the generated data__  

Once the code terminates output file named __Batches_alpha_CategoryA_CategoryB__ will be saved in  
*./Stage-2/Filter_Unbiased_Images/Filtered_Images/*  
This file contains the test data and its labels, batches of training and filter images for 3-fold cross-validation.   


