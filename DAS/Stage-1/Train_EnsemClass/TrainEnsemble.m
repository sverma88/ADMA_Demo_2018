% Function to extract data for ensemble classifier and save the parameters of the classifier   

clear all
close all
clc

%% Configuration
rfSize = 6;
numCentroids=1600;
whitening=true;
numPatches = 200000;
CIFAR_DIM=[32 32 3];
K = 7;
fixed_label = 1;
selected_labels = [(fixed_label+1):10];

%% Load CIFAR training data
fprintf('Loading training data...\n');

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


for loop_labels = 1:size(selected_labels,1)

    close all
    iter_label = selected_labels(loop_labels);
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


    %%%%%% Extract Features
    patches = zeros(numPatches, rfSize*rfSize*3);
    for i=1:numPatches
        if (mod(i,10000) == 0) fprintf('Extracting patch: %d / %d\n', i, numPatches); end

        r = random('unid', CIFAR_DIM(1) - rfSize + 1);
        c = random('unid', CIFAR_DIM(2) - rfSize + 1);
        patch = reshape(trainX(mod(i-1,size(trainX,1))+1, :), CIFAR_DIM);
        patch = patch(r:r+rfSize-1,c:c+rfSize-1,:);
        patches(i,:) = patch(:)';
    end

    % normalize for contrast
    patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

    % whiten
    if (whitening)
        C = cov(patches);
        M = mean(patches);
        [V,D] = eig(C);
        P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
        patches = bsxfun(@minus, patches, M) * P;
    end

    % run K-means
    centroids = run_kmeans(patches, numCentroids, 50);
    show_centroids(centroids, rfSize); drawnow;

    % extract training features
    if (whitening)
        trainXC = extract_features(trainX, centroids, rfSize, CIFAR_DIM, M,P);
    else
        trainXC = extract_features(trainX, centroids, rfSize, CIFAR_DIM);
    end

    %%%% Train and Test using SVM Classifier
    SVM_MDL = fitcsvm(trainXC, trainY, 'Standardize',1);

    SVM_Labels = predict(SVM_MDL, trainXC);
    fprintf('SVM training accuracy %f%%\n', 100 * (1 - sum(SVM_Labels ~= trainY) / length(trainY)));

    %%%%%% Now do it with KNN

    KNN_MDL = fitcknn(trainXC,trainY,'NumNeighbors',K,'Standardize',1);
    KNN_Labels = predict(KNN_MDL,trainXC);
        
    fprintf('k-NN training accuracy %f%%\n', 100 * (1 - sum(KNN_Labels~= trainY) / length(trainY)));
    
    %%%%%% Now do it using Naive Bayes
    NB_MDL = fitcnb(trainXC, trainY);
    Bayes_Labels = predict(NB_MDL, trainXC);

    fprintf('naive-Bayes training accuracy %f%%\n', 100 * (1 - sum(Bayes_Labels~= trainY) / length(trainY)));
 
    save(['Model_',int2str(fixed_label),'_',int2str(iter_label),'.mat'],'SVM_MDL','KNN_MDL', 'NB_MDL','centroids','M','P');
        
    
end


