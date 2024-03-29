function perceptron_letter
clc
close all

        %% Q3 PART A
        disp('=== Question 3 Part A solution is initiated. ===')

        % import and read test and train datasets
        train_images = h5read('assign1_data1.h5','/trainims');
        train_images = double(train_images)/255;
        train_labels = h5read('assign1_data1.h5','/trainlbls');
        test_images = h5read('assign1_data1.h5','/testims');
        test_images = double(test_images)/255;
        test_labels = h5read('assign1_data1.h5','/testlbls');
        
        % find size of images, number of train images and test images
        class_num = 26;
        [im_length, im_width, train_num] = size(train_images);
        [~,~,test_num] = size(test_images);
        image_per_class = train_num/class_num;

        % display sample images from each class and calculate correlation matrix
        % each class has 200 sample images
        correlation_matrix = zeros(class_num, class_num);
        figure;
        
        for class_index = 1:class_num
            %index = image_per_class*(class_index-1) + floor(image_per_class*rand) + 1; % index within 200 images
            for class_index_2 = 1:class_num
                %index_2 = image_per_class*(class_index_2-1) + floor(image_per_class*rand) + 1; % index within 200 images
                if class_index <= class_index_2 % upper triangular matrix
                    correlation_matrix(class_index, class_index_2) = corr2(train_images(:, :, image_per_class*(class_index-1)+ 32), train_images(:, :, image_per_class*(class_index_2-1)+32));
                end
            end
            subplot(5,6,class_index);
            imshow(train_images(:, :, image_per_class*(class_index-1)+ 32));
            title("Image on Class " + class_index);
        end
        correlation_matrix = correlation_matrix + correlation_matrix' - eye(class_num) % correct upper triangular matrix

        % Across class correlation for letters M and N, I and W
        figure;
        k = floor(image_per_class*rand) + 1;
        subplot(2, 2, 1)
        imshow(train_images(:, :, image_per_class*12 + k));
        title( "Class 13, Letter M");
        subplot( 2, 2, 2)
        imshow(train_images(:, :, image_per_class*13 + k));
        title( "Class 14, Letter N");
        across_correlation(1) = corr2(train_images(:, :, image_per_class*12 + k), train_images(:, :, image_per_class*13 + k));
        subplot(2, 2, 3)
        imshow(train_images(:, :, image_per_class*8 + k));
        title( "Class 9, Letter I");
        subplot( 2, 2, 4)
        imshow(train_images(:, :, image_per_class*22 + k));
        title( "Class 23, Letter W");
        across_correlation(2) = corr2(train_images(:, :, image_per_class*8 + k), train_images(:, :, image_per_class*22 + k));
        disp("Across class correlation for letters M and N, I and W is");
        across_correlation

        % Within class correlation for class 7 letter G, three images
        figure;
        subplot(2,2,[1 3])
        imshow(train_images(:, :, image_per_class*6 + 132));
        title( "Class 7, Letter M, image 132");
        subplot(2,2,2)
        imshow(train_images(:, :, image_per_class*6 + 49));
        title( "Class 7, Letter M, image 49");
        within_correlation(1) = corr2(train_images(:, :, image_per_class*6 + 132), train_images(:, :, image_per_class*6 + 49));
        subplot(2,2,4)
        imshow(train_images(:, :, image_per_class*6 + 91));
        title( "Class 7, Letter M, image 91");
        within_correlation(2) = corr2(train_images(:, :, image_per_class*6 + 132), train_images(:, :, image_per_class*6 + 91));
        disp("Within class correlation for class 7 letter G, with three images, is");
        within_correlation
      
        
        %% Q3 PART B
        disp('=== Question 3 Part B solution is initiated. ===')
        iteration = 10000;
        std = sqrt(0.01);
        mean = 0;
        input_neuron_size = im_width*im_length;
        output_neuron_num = class_num;
        %learning_rate = [1 0.2 0.02];
        learning_rate = [0.17 0.19 0.21 0.23 0.25 0.27];
        tuned = zeros(6, 2);
        
        % run the network for each learning rate
        for i = 1:6
            [MSE, weight] = perceptron(mean, std, output_neuron_num, input_neuron_size, class_num, iteration, learning_rate(i), train_images, train_num, train_labels);
            disp( "Learning Rate = " + learning_rate(i) + " MSE = " +  MSE(10000));
            tuned(i, :) = [MSE(10000) learning_rate(i)];
            w_store(i, :, :) = weight;
        end
        
        tuned_mse = tuned(  tuned(:,1) == min(tuned(:,1) ) , 1)
        tuned_learning_rate = tuned(  tuned(:,1) == min(tuned(:,1) ) , 2)
        tuned_weight = squeeze( w_store(tuned(:,1) == min(tuned(:,1)), :, 1:end-1 ) ); % size is (class_num, im_length*im_width +1)
        
        % show weights as image for each class
        figure;
        for i = 1:class_num
            subplot(5,6,i);
            tuned_weight_shaped = reshape((255*tuned_weight(i, :)), im_length, im_width); % reshape to the original image format
            imshow(tuned_weight_shaped, []);
            title("Class "+i);
        end
        
        %% Q3 PART C
        disp('=== Question 3 Part C solution is initiated. ===')
        figure;
        learning_rate_max = 2
        learning_rate_min = 0.001
        
        [MSE_tuned, weight_tuned] = perceptron(mean, std, output_neuron_num, input_neuron_size, class_num, iteration, tuned_learning_rate, train_images, train_num, train_labels);
        plot( MSE_tuned )
        hold on
        [MSE_1, weight_1] = perceptron(mean, std, output_neuron_num, input_neuron_size, class_num, iteration, learning_rate_max, train_images, train_num, train_labels);
        plot( MSE_1 )
        hold on
        [MSE_2, weight_2] = perceptron(mean, std, output_neuron_num, input_neuron_size, class_num, iteration, learning_rate_min, train_images, train_num, train_labels);
        plot( MSE_2 )
        hold on
        
        legend( "n = " + tuned_learning_rate, "n = " + learning_rate_max, "n = " + learning_rate_min)
        title( "MSE with 3 Learning Rates")
        xlabel("Iteration Number")
        ylabel("MSE")
        
        
        %% Q3 PART D
        disp('=== Question 3 Part D solution is initiated. ===')
        
        % run the network for test dataset w,th 3 different learning rates
        [accuracy_1, MSE_test_1] = perceptron_test(weight_1, test_images, test_labels, test_num, input_neuron_size , class_num);
        disp("Accuracy = %" + accuracy_1 + " when n = " + learning_rate_max);
        %disp("MSE = " + MSE_test_1);
        
        [accuracy_2, MSE_test_2] = perceptron_test(weight_2, test_images, test_labels, test_num, input_neuron_size , class_num);
        disp("Accuracy = %" + accuracy_2 + " when n = " + learning_rate_min);
        
        [accuracy_3, MSE_test_3] = perceptron_test(weight_tuned, test_images, test_labels, test_num, input_neuron_size , class_num);
        disp("Accuracy = %" + accuracy_3 + " when n = " + tuned_learning_rate);
        
        figure;
        plot(MSE_test_1)
        hold on
        plot(MSE_test_2)
        hold on
        plot(MSE_test_3)
        legend("n = " + learning_rate_max, "n = " + learning_rate_min, "n = " + tuned_learning_rate)
        title("Test MSE with different learning rates")
        xlabel("Test Index")
        ylabel("MSE")
        

end


function [MSE_test, weight_matrix_extended] = perceptron(mean, std, output_neuron_num, input_neuron_size, class_num, iteration, learning_rate, train_images, train_num, train_labels)  

weight_matrix = mean + std*randn(output_neuron_num, input_neuron_size); % random weight and bias
bias_vector = mean + std*rand(class_num, 1);
weight_matrix_extended = [weight_matrix bias_vector]; % extend it with bias

MSE_test = zeros(1, iteration);
MSE = 0;
lambda = 1;

for i=1:iteration
    
    sample_index = floor(train_num*rand())+1; %random image
    sample_image = train_images(:, :, sample_index);
    sample_input = reshape(sample_image, input_neuron_size, 1); % reshape it into vector
    
    sample_input_extended = [sample_input; -1]; % extended input
    
    potential = weight_matrix_extended * sample_input_extended; % activation potential
    
    output = 1./(1+exp(-lambda*potential)); % sigmoid activation function
    true_output =  zeros(class_num, 1);
    true_output(train_labels(sample_index))= 1; % assert the correct neurons
    
    output_error = true_output - output; % d-o
    
    mse = sum((output_error).^2) / class_num; % mean squared error for each iteration
    MSE = MSE + mse;
    MSE_test(i) = MSE/(2*i); % average the MSE
    
    % Stochastic gradient descent part
    del_weight = -output_error .* output .* (1-output) * sample_input_extended';
    
    weight_matrix_extended = weight_matrix_extended - learning_rate * del_weight;
end
end

function [accuracy, MSE_test] = perceptron_test(weight_matrix_extended, test_images, test_labels, test_num, input_neuron_size , class_num) 

lambda = 1;
MSE_test = zeros(1, test_num);
MSE = 0;
correct = 0;

for i = 1:test_num
    
    sample_image = test_images(:, :, i);
    sample_input = reshape(sample_image, input_neuron_size, 1);  % reshape it into vector
    sample_input_extended = [sample_input; -1]; % extended input
    
    potential = weight_matrix_extended * sample_input_extended; % activation potential
    
    output = 1./(1+exp(-lambda*potential)); % sigmoid activation function
    true_output =  zeros(class_num, 1);
    true_output(test_labels(i))= 1; % assert the correct neurons
    
    output_error = true_output - output; % d-o
    
    mse = sum((output_error).^2) / class_num; % mean squared error for each iteration
    MSE = MSE + mse;
    MSE_test(i) = MSE/(2*i); % average the MSE
    
    [~, arg_max] = max(output);
    decision = arg_max'; % our decision that we hope to be true
    
    if decision == test_labels(i)
        correct = correct + 1;
    end
end

accuracy = 100*correct/test_num;

end 
