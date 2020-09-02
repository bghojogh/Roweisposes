function [eigvector, eigvalue_vector] = RDA(gnd, data, r_1, r_2)
% RDA: Roweis Discriminant Analysis 
%
%       [eigvector, eigvalue] = RDA(gnd, options, data)
% 
%             Input:
%               data  - Data matrix. Each row vector of fea is a data point.
%               gnd   - Colunm vector of the label information for each
%                       data point. 
%               
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of LDA eigen-problem. 
%
%                                                   

%% make X and Y:
%---> X: column-wise data, y: labels
X = data';
y = gnd;
n_samples_training = size(X, 2);
dimensionality = size(X, 1);

%% calculate P:
if r_1 ~= 0
    K_y = calculate_delta_kernel(y);
else
    K_y = zeros(n_samples_training, n_samples_training);
end
P = (r_1 * K_y) + ((1 - r_1) * eye(n_samples_training));

%% calculate R1:
H = eye(n_samples_training) - ((1 / n_samples_training) * ones(n_samples_training, n_samples_training));
R1 = X * H * P * H * X';

%% calculate R2:
if r_2 ~= 0
    Sw = calculate_Sw(X, y);
else
    Sw = zeros(dimensionality, dimensionality);
end
R2 = (r_2 * Sw) + ((1 - r_2) * eye(dimensionality));
[eigvector, eigvalue_vector] = eig(R1, R2, 'vector');
[eigvalue_vector, ind] = sort(eigvalue_vector, 'descend');
eigvector = eigvector(:, ind);

end

function delta_kernel = calculate_delta_kernel(y)
    n_samples = length(y);
    delta_kernel = zeros(n_samples, n_samples);
    for sample_index_1 = 1:n_samples
        for sample_index_2 = 1:n_samples
            if y(sample_index_1) == y(sample_index_2)
                delta_kernel(sample_index_1, sample_index_2) = 1;
            else
                delta_kernel(sample_index_1, sample_index_2) = 0;
            end
        end
    end
end

function Sw = calculate_Sw(X, y)
    % X --> rows: features, columns: samples (column-wise)
    dimensionality = size(X, 1);
    labels_of_classes_ = sort(unique(y));
    n_classes = length(labels_of_classes_);
    X_separated_classes = separate_samples_of_classes_2(X, y);
    Sw = zeros(dimensionality, dimensionality);
    for class_index = 1:n_classes
        X_class = X_separated_classes{class_index};
        n_samples_of_class = size(X_class, 2);
        mean_of_class = mean(X_class, 2);
        X_class_centered = X_class - repmat( mean_of_class, [1,size(X_class, 2)] );
        for sample_index = 1:n_samples_of_class
            temp = X_class_centered(:, sample_index);
            Sw = Sw + (temp * temp');
        end
    end
end

function [X_separated_classes, original_index_in_whole_dataset] = separate_samples_of_classes_2(X, y)
    % it does not change the order of the samples within every class
    % X --> rows: features, columns: samples
    % return X_separated_classes --> each element of list --> rows: features, columns: samples
    labels_of_classes_ = sort(unique(y));
    n_samples = size(X, 2);
    n_dimensions = size(X, 1);
    n_classes = length(labels_of_classes_);
    X_separated_classes = cell(n_classes, 1);
    original_index_in_whole_dataset = cell(n_classes, 1);
    for class_index = 1:n_classes
        original_index_in_whole_dataset{class_index} = [];
        for sample_index = 1:n_samples
            if y(sample_index) == labels_of_classes_(class_index)
                X_separated_classes{class_index} = [X_separated_classes{class_index}, X(:, sample_index)];
                original_index_in_whole_dataset{class_index} = [original_index_in_whole_dataset{class_index}, sample_index];
            end
        end
    end
end
