%%% Code to find bias and variance and accuracy of CNN models 
clear all
clc

addpath('/data/suverma/ADMA/10/Stage-2/Train_CNN/results')
addpath('/data/suverma/ADMA/10/Stage-2/Filter_Ubiased_Images/Filtered_Images')

fixed_label = 0;
selected_labels = [fixed_label+1:9];
alpha = 10;

for loop_labels = 1:size(selected_labels,1)
    iter_label = selected_labels(loop_labels);
    
    load(['Batches_',int2str(alpha),'_',int2str(fixed_label+1),'_',int2str(iter_label+1),'.mat'])
    ground_labels = create_dist(Selected_testY);
    
    clear Selected_test Train_Batch_Images Train_Batch_Labels Unbiased_Batch_Images Unbiased_Batch_Labels  
    load(['Pred_Labels_',int2str(alpha),'_',int2str(fixed_label),'_',int2str(iter_label),'.mat'])

    m=size(Pred_Labels,1);
    pred_lab=create_dist(double(Pred_Labels));

    bi = ground_labels - pred_lab;
    va = pred_lab;

    Bias = sum(0.5*sum(bsxfun(@times,bi,bi),2))/m;
    Var = sum(0.5*(1-sum(bsxfun(@times,va,va),2)))/m;
    Acc = 1 - mean(sum(abs(double(Pred_Labels) - Selected_testY),1)/m);
    name = ['Performance_',int2str(alpha),'_',int2str(fixed_label),'_',int2str(iter_label),'.mat'];
    save(name,'Bias','Var','Acc');
    
end

        
