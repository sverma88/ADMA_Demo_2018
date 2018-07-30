% Function to classify data points using K- nearest neighbor classifier


function [Batch_Images,Batch_Labels]=Create_Batches(Images,Labels)

%Input
%
% Images                   : Matrix containing  Images 
% Labels                   : Corresponding Labesl of Images
%
% Output
% Batch_Images             : Images as batches
% Batch_Labels             : Labels as batches
% 
% 
% Author                   : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update              : 08/10/2017
%
% 

rng(5);

Total_Classes = 2;
cross_valid = 3;

Batched_Img = cell(Total_Classes,cross_valid);
Batched_Lab = cell(Total_Classes,cross_valid);


for i = 1:Total_Classes
    Ind = find(Labels == i);
    if(size(Ind,1)~=0)

        K_indices = crossvalind('kfold', size(Ind,1),cross_valid);

        for j=1:cross_valid
            if(size(K_indices,1) < cross_valid)
                Img = Images(Ind,:);
                Lab = Labels(Ind,:);
                Batched_Img{i,j} = Img;
                Batched_Lab{i,j} = Lab;
            else
                rm_index = find(K_indices == j);
                Img = Images(Ind,:);
                Lab = Labels(Ind,:);
                Img(rm_index,:) = [];
                Lab(rm_index,:) = [];
                Batched_Img{i,j} = Img;
                Batched_Lab{i,j} = Lab;
            end
        end
    end

end
 
Batch_Images = cell(1,cross_valid);
Batch_Labels = cell(1,cross_valid);

for i = 1:cross_valid
    
    B_Img = [];
    B_Lab = [];
    
    for j = 1:Total_Classes
        
        B_Img = [B_Img ; Batched_Img{j,i}];
        B_Lab = [B_Lab ; Batched_Lab{j,i}];
    end
    
    Batch_Images{1,i} =  B_Img;
    Batch_Labels{1,i} =  B_Lab;
end

end
       
        