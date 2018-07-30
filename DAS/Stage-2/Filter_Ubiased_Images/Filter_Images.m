% Function to filter the synthetic images   

clear all
close all
clc

%% Configuration
rfSize = 6;
whitening=true;
CIFAR_DIM=[32 32 3];
fixed_label = 1;
selected_labels = [(fixed_label+1):10];

%% Load CIFAR training data
%%% add path to the saved ensembled classifier model
addpath('/data/suverma/ADMA/10/Stage-1/Train_EnsemClass')
addpath('/data/suverma/ADMA/10/Stage-1/Train_GAN/CIFAR-10/');

load('Train_Images.mat');
Images = permute(Train_Images, [1,4,2,3]);

Train_Data = zeros(50000,3072);
for i=1:50000
    R = reshape(Images(i,1,:,:),32,32);
    G = reshape(Images(i,2,:,:),32,32);
    B = reshape(Images(i,3,:,:),32,32);
    R1 = reshape(R',1,[]);
    G1 = reshape(G',1,[]);
    B1 = reshape(B',1,[]);
    Train_Data(i,:)=[R1 G1 B1];
end

load('Train_Labels.mat');
Train_Labels = double(Train_Labels)' + 1;

clear Images Train_Images

trainX = Train_Data;
trainY = Train_Labels;
rng(5)
random_index = randperm(size(trainX,1));
    
trainX_whole = trainX(random_index, :);
trainY_whole = trainY(random_index, :);


%%%%%% Load the whole test data

load('Test_Images.mat');
Images = permute(Test_Images, [1,4,2,3]);

Test_Data = zeros(10000,3072);
for i=1:10000
    R = reshape(Images(i,1,:,:),32,32);
    G = reshape(Images(i,2,:,:),32,32);
    B = reshape(Images(i,3,:,:),32,32);
    R1 = reshape(R',1,[]);
    G1 = reshape(G',1,[]);
    B1 = reshape(B',1,[]);
    Test_Data(i,:)=[R1 G1 B1];
end

load('Test_Labels.mat');
Test_Labels = double(Test_Labels)' + 1;

clear Images Test_Images

testX = Test_Data;
testY = Test_Labels;
rng(5)
random_index = randperm(size(testX,1));
    
testX_whole = testX(random_index, :);
testY_whole = testY(random_index, :);


fprintf('Loading Synthetic Generated Data...\n');
addpath('/data/suverma/ADMA/10/Stage-1/Train_GAN/Generated_Data/');

write_path = ['/data/suverma/ADMA/10/Stage-2/Filter_Ubiased_Images/Filtered_Images/'];


for loop_labels = 1:size(selected_labels,1)
   
   iter_label = selected_labels(loop_labels);
   load (['Model_',int2str(fixed_label),'_',int2str(iter_label),'.mat']);
   
   
   %%%% extract the train subset data and create batches   
   selected_index1 = find(trainY_whole == fixed_label);
   selected_index2 = find(trainY_whole == iter_label);
   
   Selected_train1 = trainX_whole(selected_index1,:);
   Selected_train2 = trainX_whole(selected_index2,:);
   
   Selected_train = [Selected_train1; Selected_train2];
   
   Selected_trainY1 = trainY_whole(selected_index1, :);
   Selected_trainY2 = trainY_whole(selected_index2, :);
   
   Selected_trainY = [ones(size(Selected_trainY1)); 2*ones(size(Selected_trainY2))];
   
   trainX = Selected_train;
   trainY = Selected_trainY;
   
 
   [Train_Batch_Images, Train_Batch_Labels] = Create_Batches(trainX, trainY);
   
   
   %%%%% Extract Test Data and Create Batches
   
   selected_index1 = find(testY_whole == fixed_label);
   selected_index2 = find(testY_whole == iter_label);
   
   Selected_test1 = testX_whole(selected_index1,:);
   Selected_test2 = testX_whole(selected_index2,:);
   
   Selected_test = [Selected_test1; Selected_test2];
   
   Selected_testY1 = trainY_whole(selected_index1, :);
   Selected_testY2 = trainY_whole(selected_index2, :);
   
   Selected_testY = [ones(size(Selected_testY1)); 2*ones(size(Selected_testY2))];

   %%%% Filter the synthetic Images and create batches   
   
   for loop_alpha=10:10:90      
               
        load(['Images_',int2str(loop_alpha),'_',int2str(fixed_label-1),'_',int2str(iter_label-1),'.mat'])
        
        Test_Data = zeros(size(Images,1),3072);
        Images = permute(Images, [1,4,2,3]);
        
        for i=1:size(Images,1)
            R = reshape(Images(i,1,:,:),32,32);
            G = reshape(Images(i,2,:,:),32,32);
            B = reshape(Images(i,3,:,:),32,32);
            R1 = reshape(R',1,[]);
            G1 = reshape(G',1,[]);
            B1 = reshape(B',1,[]);
            Test_Data(i,:)=[R1 G1 B1];
        end
        
        
        load(['Labels_',int2str(loop_alpha),'_',int2str(fixed_label-1),'_',int2str(iter_label-1),'.mat'])
        Test_Labels = sum(bsxfun(@times, double(Labels), 0:1), 2)+1;
              
        %%%%% Prepare TEST Data %%%%%
        
        %% Load CIFAR test data
        fprintf('Loading test data...\n');
        testX = Test_Data;
        testY = Test_Labels;
        
        % compute testing features and standardize
        if (whitening)
            testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM, M,P);
        else
            testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM);
        end
        
        SVM_Labels = predict(SVM_MDL, testXC);
        fprintf('SVM Test accuracy %f%%\n', 100 * (1 - sum(SVM_Labels~= testY) / length(testY)));
              
        KNN_Labels=predict(KNN_MDL,testXC);
        
        fprintf('KNN Test accuracy %f%%\n', 100 * (1 - sum(KNN_Labels~= testY) / length(testY)));
        
        Bayes_Labels = predict(NB_MDL, testXC);
        
        fprintf('Bayes Test accuracy %f%%\n', 100 * (1 - sum(Bayes_Labels~= testY) / length(testY)));

        All_Labels = [SVM_Labels KNN_Labels Bayes_Labels testY];
        
        equal_labels = std(All_Labels', 1);
        
        Correct_indices= find(equal_labels' == 0);
        
        Unbiased_Images = Test_Data(Correct_indices, :);
        Unbiased_Labels = Test_Labels(Correct_indices, :);
        
        [Unbiased_Batch_Images, Unbiased_Batch_Labels] = Create_Batches(Unbiased_Images, Unbiased_Labels);
        
        name = [write_path,'Batches_',int2str(loop_alpha),'_',int2str(fixed_label),'_',int2str(iter_label),'.mat'];
        
        save(name, 'Unbiased_Batch_Images', 'Unbiased_Batch_Labels', 'Selected_test', 'Selected_testY', 'Train_Batch_Images', 'Train_Batch_Labels');
        
        clear All_Labels Unbiased_Images Unbiased_Labels SVM_Labels KNN_Labels Bayes_labels
        
    end

end


