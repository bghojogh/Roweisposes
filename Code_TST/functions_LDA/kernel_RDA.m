function [eigvector, eigvalue_vector] = kernel_RDA(gnd, data, r_1, r_2, kernel_type, kernel_parameters)
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

%% calculate M:
K_x = pairwise_kernel(X', X', kernel_type, kernel_parameters);
H = eye(n_samples_training) - ((1 / n_samples_training) * ones(n_samples_training, n_samples_training));
M = K_x * H * P * H * K_x;

%% calculate L:
if r_2 ~= 0
    N = calculate_N(X, y, kernel_type, kernel_parameters);
else
    N = zeros(n_samples_training, n_samples_training);
end

% N = ProjectOntoPositiveSemideinite(N);  %--> make it positive semi-definite
N = N + (0.01 * eye(size(N,1)));  %--> make it full rank and invertibale
% N = 0.5 * (N + N');  %--> make it symmetric

L = (r_2 * N) + ((1 - r_2) * K_x);

% L = 0.5 * (L + L');
% L = ProjectOntoPositiveSemideinite(L);

% L = L + (0.01 * eye(size(L,1)));

% [eigvector, eigvalue_vector] = eig(M, L, 'vector');
number_of_eigenvectors = 15;
% [eigvector, eigvalue_matrix] = eigs(M, L, number_of_eigenvectors);

% W = (L + (10^6 * eye(size(L,1)))) \ M;
W = L \ M;
[eigvector, eigvalue_matrix] = eigs(W, number_of_eigenvectors);

eigvalue_vector = diag(eigvalue_matrix);
[eigvalue_vector, ind] = sort(eigvalue_vector, 'descend');
eigvector = eigvector(:, ind);

end

function [A] = ProjectOntoPositiveSemideinite(A)

    [V, D] = eig(A);
    D(D < 0) = 0;
    A = V * D * transpose(V);

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

function N = calculate_N(X, y, kernel_type, kernel_parameters)
    % X --> rows: features, columns: samples (column-wise)
    n_samples_training = size(X, 2);
    labels_of_classes_ = sort(unique(y));
    n_classes = length(labels_of_classes_);
    X_separated_classes = separate_samples_of_classes_2(X, y);
    N = zeros(n_samples_training, n_samples_training);
    for class_index = 1:n_classes
        X_class = X_separated_classes{class_index};
        n_samples_of_class = size(X_class, 2);
        K_all_and_class = pairwise_kernel(X', X_class', kernel_type, kernel_parameters);
        H_class = eye(n_samples_of_class) - ((1 / n_samples_of_class) * ones(n_samples_of_class, n_samples_of_class));
        temp_ = (K_all_and_class * H_class * K_all_and_class');
        temp_ = temp_ + (0.01 * eye(size(temp_,1)));
        temp_ = 0.5 * (temp_ + temp_');  %--> make it symmetric
        N = N + temp_;
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
