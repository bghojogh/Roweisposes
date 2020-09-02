function [u_norm_LDA, data_LDA] = Train_LDA(test_person, method_and_options)

global index_of_states;
global number_of_states;
global number_of_state_samples;
global state;
global name_of_states;
global name_of_actions;
global joints_selected;
global number_of_selected_joints;
global do_subtract_from_total_mean;
global display_projected_LDA_mode;
global LDA_projected_joints_of_specific_states;
global project_of_means_MODE;
global distance_type;
global const_cov;
global use_manual_mahalanobis;
global LDA_projected_state_means;
global projected_means_of_classes;
global LDA_projected_states;
global covariance_matrix;
global distance_average_of_averages;
global confusion_matrix_trainStates;
global confusion_matrix_trainStates_minDistances;
global display_eigenvectors;

%% preparing LDA data:
joints_of_specific_states = cell(number_of_states,1);  % each cell contains a set of joints of one state. in each cell: 1st dimension is joints, 2nd dimension is (x,y,z), 3rd dimension is samples of states
for state_index = 1:number_of_states
    joints_of_specific_states{state_index} = zeros(number_of_selected_joints,3,[]);
end
counter = 0;
for i = 1:number_of_state_samples
    if state(i,1) ~= test_person
        counter = counter + 1;

        which_person = state(i,1); which_performance = state(i,2); which_action = state(i,3); frame = state(i,4);
        joints = load_joints_and_align_them(which_person,which_performance,which_action,frame);

        joints_stack(:,:,counter) = joints;
        %%%% preparing LDA inputs:
        data_LDA(counter,:) = reshape(joints_stack(:,:,counter),1,[]);
        labels_LDA(counter,1) = state(i,5);
        joints_of_specific_states{labels_LDA(counter,1)}(:,:,end+1) = joints_stack(:,:,counter);
    end
end

%% Train LDA:
train_joints_total_mean = mean(joints_stack,3);
train_joints_state_mean = zeros(number_of_selected_joints,3,number_of_states);
for state_index = 1:number_of_states
    train_joints_state_mean(:,:,state_index) = mean(joints_of_specific_states{state_index}, 3);
end
if strcmp(method_and_options{1}, 'LDA')
    options = [];   
    [V,~] = LDA(labels_LDA,options,data_LDA);
elseif strcmp(method_and_options{1}, 'RDA')
    r_1 = method_and_options{2}(1);
    r_2 = method_and_options{2}(2);
    [V,~] = RDA(labels_LDA, data_LDA, r_1, r_2);
    V = V(:, 1:(number_of_states-1));
elseif strcmp(method_and_options{1}, 'kernel_RDA')
    r_1 = method_and_options{2}(1);
    r_2 = method_and_options{2}(2);
    kernel_type = method_and_options{3};
    kernel_parameters = method_and_options{4};
    [V,~] = kernel_RDA(labels_LDA, data_LDA, r_1, r_2, kernel_type, kernel_parameters);
    V = V(:, 1:(number_of_states-1));
end
u_norm_LDA = V';

%% Plot the eigenvectors (Fisher directions):
if display_eigenvectors == 1
    line_width = 1;
    for class = 1:3   %number_of_states-1
        figure
        view(3)
        joints = reshape(u_norm_LDA(class,:), [], 3);
        joints = [joints(1:2,:); 0,0,0; joints(3:end,:)];
        hold on
        MarkerSize = 7;
        joint_index = find(joints_selected == 1); plots(1) = plot3(joints(joint_index,1),joints(joint_index,3),joints(joint_index,2),'ob', 'MarkerFaceColor', 'b', 'MarkerSize', MarkerSize);
        joint_index = find(joints_selected == 9); plots(2) = plot3(joints(joint_index,1),joints(joint_index,3),joints(joint_index,2),'sb', 'MarkerFaceColor', 'b', 'MarkerSize', MarkerSize);
        joint_index = find(joints_selected == 6); plots(3) = plot3(joints(joint_index,1),joints(joint_index,3),joints(joint_index,2),'db', 'MarkerFaceColor', 'b', 'MarkerSize', MarkerSize);
        joint_index = find(joints_selected == 15); plots(4) = plot3(joints(joint_index,1),joints(joint_index,3),joints(joint_index,2),'>b', 'MarkerFaceColor', 'b', 'MarkerSize', MarkerSize);
        joint_index = find(joints_selected == 12); plots(5) = plot3(joints(joint_index,1),joints(joint_index,3),joints(joint_index,2),'<b', 'MarkerFaceColor', 'b', 'MarkerSize', MarkerSize);
        leg = legend(plots,'Head','Right hand','Left hand','Right leg','Left leg');
        for i = 1:5
            switch i
                case 1
                    joint_first = find(joints_selected == 1);
                    joint_second = find(joints_selected == 9);
                case 2
                    joint_first = find(joints_selected == 9);
                    joint_second = find(joints_selected == 15);
                case 3
                    joint_first = find(joints_selected == 15);
                    joint_second = find(joints_selected == 12);
                case 4
                    joint_first = find(joints_selected == 12);
                    joint_second = find(joints_selected == 6);
                case 5
                    joint_first = find(joints_selected == 6);
                    joint_second = find(joints_selected == 1);
            end
            plot3([joints(joint_first,1), joints(joint_second,1)],...
                  [joints(joint_first,3), joints(joint_second,3)],...
                  [joints(joint_first,2), joints(joint_second,2)],...
                  '-r', 'LineWidth', line_width);
        end
    %     xlim([min(joints(:,1))*10, max(joints(:,1))*10]);
    %     ylim([min(joints(:,3))*10, max(joints(:,3))*10]);
    %     zlim([min(joints(:,2))*1.5, max(joints(:,2))*2]);
        xlim([-0.15, 0.17]);
        ylim([-0.35, 0.2]);
        zlim([-0.3, 0.3]);
        xlabel('x'); ylabel('z'); zlabel('y');
        str = sprintf('Eigenvector: %d', class);
        title(str);
        hold off
        set(gcf,'units','normalized','outerposition',[0 0 1 1]);  % full sreen
    end
    return
end

%% projecting mean of states onto the Fisher LDA projection directions:
LDA_projected_state_means = zeros(size(u_norm_LDA,1),1,number_of_states);
for state_index = 1:number_of_states
    if strcmp(method_and_options{1}, 'LDA') || strcmp(method_and_options{1}, 'RDA')
        if do_subtract_from_total_mean == 1
            LDA_projected_state_means(:,:,state_index) = u_norm_LDA * reshape(train_joints_state_mean(:,:,state_index) - train_joints_total_mean,[],1);
        else
            LDA_projected_state_means(:,:,state_index) = u_norm_LDA * reshape(train_joints_state_mean(:,:,state_index),[],1);
        end
    elseif strcmp(method_and_options{1}, 'kernel_RDA')
        kernel_type = method_and_options{3};
        kernel_parameters = method_and_options{4};
        the_kernel = pairwise_kernel(data_LDA, reshape(train_joints_state_mean(:,:,state_index),1,[]), kernel_type, kernel_parameters);
        LDA_projected_state_means(:,:,state_index) = u_norm_LDA * the_kernel;
    end
end

%% projecting state samples onto the Fisher LDA projection directions:
LDA_projected_joints_of_specific_states = cell(number_of_states,1);  % each cell contains a set of joints of one state. in each cell: 1st and 2nd dimensions are projected joints, 3rd dimension is samples of states
for state_index = 1:number_of_states
    LDA_projected_joints_of_specific_states{state_index} = zeros(size(u_norm_LDA,1),1,size(joints_of_specific_states{state_index},3));
end
for state_index = 1:number_of_states
    number_of_samples_of_states = size(joints_of_specific_states{state_index}(:,:,:),3);
    for state_sample = 1:number_of_samples_of_states
        if strcmp(method_and_options{1}, 'LDA') || strcmp(method_and_options{1}, 'RDA')
            if do_subtract_from_total_mean == 1
                LDA_projected_joints_of_specific_states{state_index}(:,:,state_sample) = u_norm_LDA * reshape(joints_of_specific_states{state_index}(:,:,state_sample) - train_joints_total_mean,[],1);
            else
                LDA_projected_joints_of_specific_states{state_index}(:,:,state_sample) = u_norm_LDA * reshape(joints_of_specific_states{state_index}(:,:,state_sample),[],1);
            end
        elseif strcmp(method_and_options{1}, 'kernel_RDA')
            kernel_type = method_and_options{3};
            kernel_parameters = method_and_options{4};
            the_kernel = pairwise_kernel(data_LDA, reshape(joints_of_specific_states{state_index}(:,:,state_sample),1,[]), kernel_type, kernel_parameters);
            LDA_projected_joints_of_specific_states{state_index}(:,:,state_sample) = u_norm_LDA * the_kernel;
        end
    end
end

%% display the two/three first dimensions of LDA-projected samples:
if display_projected_LDA_mode == 1
    disp('Plot mode: Plotting the projected states onto Fisher directions...');
    plot_LDA_projected_states(LDA_projected_joints_of_specific_states, number_of_states, name_of_states, LDA_projected_state_means);
    return  %%% terminate the program
end

%% creatng LDA_projected_states (needed for mahalanobis):
for state_index = 1:number_of_states
    number_of_samples_of_states = size(LDA_projected_joints_of_specific_states{state_index}(:,:,:),3);
    for state_sample = 1:number_of_samples_of_states
        %%%% LDA_projected_states: cell array (each cell is about a state)
        %%%% LDA_projected_states{state_index}: rows are state samples, columns are reshaped horizontal vector of "LDA_projected_joints_of_specific_states"
        LDA_projected_states{state_index}(state_sample,:) = reshape(LDA_projected_joints_of_specific_states{state_index}(:,:,state_sample),1,[]); 
    end
end

%% calculating: "projection of means of classes" + "covariance matices for mahalanobis":
%%%% means:
for state_index = 1:number_of_states
    if project_of_means_MODE == 1
        %%%% project of means:
        projected_means_of_classes(state_index,:) = reshape(LDA_projected_state_means(:,:,state_index),1,[]);
    elseif project_of_means_MODE == 2
        %%%% means of projected samples:
        projected_means_of_classes(state_index,:) = mean(LDA_projected_states{state_index}(:,:));
    end
end
%%%% covariance matrices:
if distance_type == 2   %--> only needed for mahalanobis distance:
    for state_index = 1:number_of_states
        covariance_matrix(:,:,state_index) = cov(LDA_projected_states{state_index}(:,:));
    end
else
    covariance_matrix = [];  %--> not important for Euclidean distance
end

%% calculating the average distance of train samples from the means of their class (needed for threshold in window):
for state_index = 1:number_of_states
    number_of_samples_of_states = size(joints_of_specific_states{state_index}(:,:,:),3);
    distance = 0;
    for state_sample = 1:number_of_samples_of_states
        projected_sample = (LDA_projected_joints_of_specific_states{state_index}(:,:,state_sample))';
        if distance_type == 1  %%% Euclidean distance
            projected_mean_of_class = reshape(LDA_projected_state_means(:,:,state_index),1,[]);
            distance = distance + Euclidean_distance(projected_sample, projected_mean_of_class);
        elseif distance_type == 2  %%% Mahalanobis distance
            if use_manual_mahalanobis == 1
                distance = distance + mahalanobis(projected_sample, projected_means_of_classes(state_index,:), covariance_matrix(:,:,state_index), const_cov);
            else
                distance = distance + mahal(projected_sample, LDA_projected_states{state_index}(:,:));
            end
        end
    end
    distance_average(state_index) = distance / number_of_samples_of_states;
end
distance_average_of_averages = mean(distance_average);

%% confusion matrix of train states:
for i = 1:number_of_state_samples

    %%%% loading skeleton:
    which_person = state(i,1); which_performance = state(i,2); which_action = state(i,3); frame = state(i,4);
    joints = load_joints_and_align_them(which_person,which_performance,which_action,frame);

    %%%% projecting state samples onto the Fisher LDA projection directions:
    if strcmp(method_and_options{1}, 'LDA') || strcmp(method_and_options{1}, 'RDA')
        if do_subtract_from_total_mean == 1
            LDA_projected_joints = u_norm_LDA * reshape(joints - train_joints_total_mean,[],1);
        else
            LDA_projected_joints = u_norm_LDA * reshape(joints,[],1);
        end
    elseif strcmp(method_and_options{1}, 'kernel_RDA')
        kernel_type = method_and_options{3};
        kernel_parameters = method_and_options{4};
        the_kernel = pairwise_kernel(data_LDA, reshape(joints,1,[]), kernel_type, kernel_parameters);
        LDA_projected_joints = u_norm_LDA * the_kernel;
    end

    %%%% distance of projected joints and projected mean of states:
    [estimated_class, distance_estimated_class, distances] = calculate_distance(LDA_projected_joints, LDA_projected_state_means, projected_means_of_classes, LDA_projected_states, covariance_matrix, const_cov, distance_type, use_manual_mahalanobis, number_of_states);

    %%%% confusion matrix:
    actual_state = state(i,5);
    estimated_state = estimated_class;
    confusion_matrix_trainStates(actual_state,estimated_state,test_person) = confusion_matrix_trainStates(actual_state,estimated_state,test_person) + 1;
    confusion_matrix_trainStates_minDistances(actual_state,estimated_state,test_person) = confusion_matrix_trainStates_minDistances(actual_state,estimated_state,test_person) + distance_estimated_class;

end
confusion_matrix_trainStates_minDistances(:,:,test_person) = confusion_matrix_trainStates_minDistances(:,:,test_person) ./ confusion_matrix_trainStates(:,:,test_person); %--> averaging the matrix of distance
confusion_matrix_trainStates_minDistances(isnan(confusion_matrix_trainStates_minDistances)) = 0; %--> remove NaNs
distance_estimated_class = [];  %--> make empty for further use

end
