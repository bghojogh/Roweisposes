function [E,P,Pi,Threshold] = Train_the_HMM(test_person, u_norm_LDA, method_and_options, data_LDA)

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
global action_length;
global frame_step;
global number_of_actions;
global number_of_persons;
global number_of_samples;
global report_trained_states_mode;
global report_trained_states_mode_ACTION;
global step_of_sliding;
global length_of_window;
global INITIAL_distance_factor_for_accepting_frame;
global STEP_distance_factor_for_accepting_frame;
global do_HMM_equal_sequence_numbers;

maximum_frame_length = max(max(max(action_length)));
sequence_length = floor(maximum_frame_length / frame_step) + 1;  % number of intervals is "floor(maximum_frame_length / frame_step)", so sequence_length is "floor(maximum_frame_length / frame_step) + 1"
HMM_labels = cell(number_of_actions,1);  %%% preallocation - part 1
for which_person = 1:number_of_persons
    if which_person ~= test_person
        for which_action = 1:number_of_actions
            number_of_performances = number_of_samples(which_person,which_action);
            for which_performance = 1:number_of_performances
                %%%% report trained states:
                if report_trained_states_mode == 1
                    if which_action == report_trained_states_mode_ACTION
                        disp('=============================');
                        str = sprintf('Actual Action: %s', name_of_actions{which_action});
                        disp(str);
                        str = sprintf('Train Person: %d', which_person);
                        disp(str);
                        str = sprintf('Performance: %d', which_performance);
                        disp(str);
                        disp('Trained States:');
                    end
                end
                %%%% finding distances:
                frames = 1:frame_step:action_length(which_person,which_performance,which_action);
                for frame = frames
                    %%%% load joints:
                    joints = load_joints_and_align_them(which_person,which_performance,which_action,frame);
                    %%%% projecting frame samples onto the Fisher LDA projection directions:
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
                    %%%% distance between projected joints and classes:
                    [estimated_class(frame), distance_estimated_class(frame), ~] = calculate_distance(LDA_projected_joints, LDA_projected_state_means, projected_means_of_classes, LDA_projected_states, covariance_matrix, const_cov, distance_type, use_manual_mahalanobis, number_of_states);
                end
                %%%% window:
                frames = 1:frame_step:action_length(which_person,which_performance,which_action);
                counter_of_slidings = 1;
                finished_sliding_window = 0;
                last_accepted_frame_in_previous_window = -1;  %--> should not be considered in first window
                while ~finished_sliding_window
                    index_first_of_window = (counter_of_slidings - 1)*step_of_sliding + 1;
                    index_last_of_window = (counter_of_slidings - 1)*step_of_sliding + length_of_window;
                    if index_last_of_window <= length(frames)
                        frames_in_window = frames(index_first_of_window : index_last_of_window);
                    else
                        frames_in_window = frames(index_first_of_window : end);
                        finished_sliding_window = 1;
                    end
                    if index_first_of_window > length(frames)
                        break;
                    end
                    is_all_empty_in_window = 1;
                    distance_factor_for_accepting_frame = INITIAL_distance_factor_for_accepting_frame;
                    counter_loop = 0;
                    while is_all_empty_in_window
                        counter_loop = counter_loop + 1;
                        if counter_loop >= 1000 %--> has taken too many iterations
                            temp_to_be_copied_in_future = 1;
                            break
                        end
                        for frame = frames_in_window
                            estimated_class_this_frame = estimated_class(frame);
                            distance_estimated_class_this_frame = distance_estimated_class(frame);
                            if distance_estimated_class_this_frame > distance_factor_for_accepting_frame + distance_average_of_averages
                                do_count_this_frame = 0;
                            else
                                if last_accepted_frame_in_previous_window == frames(end)  %--> if last accepted frame was the last frame
                                    do_count_this_frame = 0;     % do not count this frame because all needed frames are selected
                                    is_all_empty_in_window = 0;  % do not continue the loop because all needed frames are selected
                                else
                                    if frame <= last_accepted_frame_in_previous_window
                                        do_count_this_frame = 0;
                                    else
                                        do_count_this_frame = 1;
                                        is_all_empty_in_window = 0;  % at least one frame is considered and thus window is not empty
                                        temp_to_be_copied_in_future = frame;
                                    end
                                end
                            end

                            index_of_total_performances = 0;
                            for iii = 1:which_person-1
                                if iii ~= test_person
                                    index_of_total_performances = index_of_total_performances + number_of_samples(iii,which_action);
                                end
                            end
                            index_of_total_performances = index_of_total_performances + which_performance;
                            if do_count_this_frame == 1
                                if isempty(HMM_labels{which_action})
                                    l = 0;
                                elseif index_of_total_performances > length(HMM_labels{which_action})
                                    l = 0;
                                else
                                    l = length(HMM_labels{which_action}{index_of_total_performances});
                                end
                                HMM_labels{which_action}{index_of_total_performances}(l + 1) = estimated_class_this_frame;  %% --> dimension 1 of cell: action, dimension 2 of cell: performance (total), dimension 3 (array): frames
                            end
                            %%%% report trained states:
                            if report_trained_states_mode == 1
                                if which_action == report_trained_states_mode_ACTION
                                    str = sprintf('>>>>>> state %d: %s', estimated_class_this_frame, name_of_states{estimated_class_this_frame});
                                    disp(str);
                                end
                            end

                        end
                        distance_factor_for_accepting_frame = distance_factor_for_accepting_frame + STEP_distance_factor_for_accepting_frame;
                    end
                    counter_of_slidings = counter_of_slidings + 1;
                    last_accepted_frame_in_previous_window = temp_to_be_copied_in_future;
                end

                %%%% labeling the remaining frames (till the maximum_frame_length) as the last found state:
                if do_HMM_equal_sequence_numbers == 1
                    for fend = (f+1) : sequence_length
                        HMM_labels{which_action}{end}(fend) = HMM_labels{which_action}{end}(f);
                    end
                end
            end
        end
    end
end
%%%% Train HMM:
%     for index_M = 1:number_of_actions
%         M(index_M,1) = 6;
%     end
%M=[3;3;3;3;3;3;3;3;3;3];   % the length of M is number_of_actions
%M=[2;4;2;3;4;2;3;3;3;4];   % the length of M is number_of_actions
% M=[3;3;4;2;2;2;3;3;2;5];   % the length of M is number_of_actions

M = [2;2;2;2;2;2;2;2;2;2];

[E,P,Pi,Threshold] = train_HMM(HMM_labels,number_of_states,M,number_of_actions);

end