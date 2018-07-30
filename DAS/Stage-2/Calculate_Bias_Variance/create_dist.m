% % function to convert interger matrix to one-hot encoding 
% % and add N columns of this matrix 

function [One_hot] = create_dist(Index_matrix)

% Input
% Index_Matrix  : M*N Matrix, element in [1-10]
% 
% Output
% One_hot       : M*10 Matrix, element is avg(one hot vectors)

Index_matrix = Index_matrix-1;

[m,n] = size(Index_matrix);
n_classes=2;

y_one_hot=zeros(m,n_classes);

Index_matrix = Index_matrix + ones(size(Index_matrix));

for j=1:n
    y=Index_matrix(:,j);
    y_hot=zeros(m,n_classes);
    for i = 1:n_classes
        rows = y == i;
        y_hot( rows, i ) = 1;
    end
    y_one_hot=y_one_hot + y_hot;
end

One_hot = y_one_hot/n;


