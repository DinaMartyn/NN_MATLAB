clear;
close all;

activation_function=@(a) a;
activation_function_derivative=@(a) 1;

testfunktion=@(x) sin(x(1)/10);

% Number of neurons in every layer in the form
% [number of inputs, number of neurons in the first hidden layer, ...., number of neurons in the last hidden layer, number of outputs]
number_NN = [1 3 3 1];
number_of_NN = length(number_NN)-1;

learning_rate = 0.2;

%% NN as an array of structures
NN_layer(1:number_of_NN) = struct('inp',0,'w',0,'alpha',0,'delta',0,'outp',0);
% Waighting initialization
for i = 1:number_of_NN
    NN_layer(i).w = 2*rand(number_NN(i+1),number_NN(i))-1;
end


%%
number_of_samples = 1000;
data_range = [-0.8,0.8];

samples = zeros(number_NN(1), number_of_samples);
d = zeros(number_NN(end), number_of_samples);
for i = 1:number_of_samples
    samples(:,i) = (abs(data_range(1)) + abs(data_range(2)))*rand(number_NN(1),1) + data_range(1);
    d(:,i) = testfunktion(samples(:,i));
end

%% Learning
for numb_iter = 1:125

    sample_test_id = randi(number_of_samples);
    sample_i = samples(:,sample_test_id);
    d_i = d(:,sample_test_id);
    
    %% Forward calculation
    for j = 1:number_of_NN
        if j < 2
            NN_layer(j).inp = sample_i;
        else
            NN_layer(j).inp = NN_layer(j-1).outp;
        end
        NN_layer(j).alpha = NN_layer(j).w*NN_layer(j).inp;
        NN_layer(j).outp = activation_function(NN_layer(j).alpha);
    end
    
    %% Backward calculation
    error = d_i - NN_layer(number_of_NN).outp;

    %% local gradients of the neurons
    NN_layer(number_of_NN).delta = error.*activation_function_derivative(NN_layer(number_of_NN).alpha);
    
    for i = number_of_NN:-1:2
        layer_length = size(NN_layer(i).w,2);
        NN_layer(i-1).delta = zeros(layer_length,1);
        for k = 1:layer_length
            NN_layer(i-1).delta(k) = sum(NN_layer(i).w(:,k).* NN_layer(i).delta)*activation_function_derivative(NN_layer(i-1).alpha(k)); 
        end
    end
    
    %% Update of weights
    for i = number_of_NN:-1:1
        NN_layer(i).w = NN_layer(i).w + learning_rate*NN_layer(i).delta*NN_layer(i).inp';
    end

end
% Verification

data_range = [-0.8,0.8];
step = 0.1;
X = data_range(1):step:data_range(2);
d = zeros(size(X));
NN_y = zeros(size(X));
error = zeros(1, length(X));
k = 1;
for i = 1:length(X)  

        d(i) = testfunktion(X(i));

        %% Forward NN calculation
        for j = 1:number_of_NN
            if j < 2
                NN_layer(j).inp = X(i);
            else
                NN_layer(j).inp = NN_layer(j-1).outp;
            end
            NN_layer(j).alpha = NN_layer(j).w*NN_layer(j).inp;
            NN_layer(j).outp = activation_function(NN_layer(j).alpha);
        end
        %%
        NN_y(i) = NN_layer(j).outp;
        error(k) = norm(d(i) - NN_y(i));
        k = k +1;
    
end


%% Illustrare the results
% fig = figure;
% set(fig, 'units', 'inches', 'position', [1 0 8 4]);
% 
% hold on 
% title('Neural Net');
% for i = 1:number_of_NN
%     layer_length = size(NN_layer(i).inp,1);
%     for k = 1:layer_length
%         plot(i,k-layer_length/2,'o','MarkerSize',20,'MarkerEdgeColor','k'); 
%         str = sprintf('%.2f', NN_layer(i).inp(layer_length-k+1));
%         text(i,k-layer_length/2+0.25, str); 
%     end
%     if i == number_of_NN
%         layer_length = size(NN_layer(i).outp,1);
%         for k = 1:layer_length
%             plot(i+1,k-layer_length/2,'o','MarkerSize',20,'MarkerEdgeColor','k'); 
%             str = sprintf('%.2f', NN_layer(i).outp(layer_length-k+1));
%             text(i+1,k-layer_length/2+0.25, str);
%         end
%     end
% end
% 
% for i = 1:number_of_NN
%     for k = 1:length(NN_layer(i).inp)
%         for l = 1:length(NN_layer(i).outp)
%             axis(axis)
%             arrow([i,k-length(NN_layer(i).inp)/2], [i+1,l-length(NN_layer(i).outp)/2]);
%             str = sprintf('%.2f', NN_layer(i).w(length(NN_layer(i).outp)-l+1,length(NN_layer(i).inp)-k+1));
% 
%             x = i + 0.25;
%             y1 = k - length(NN_layer(i).inp)/2 ;
%             y2 = l - length(NN_layer(i).outp)/2;
%             y = (y2 - y1)/4 + y1;
%             text(x,y, str);
%         end
%     end
% end

fig = figure;
set(fig, 'units', 'inches', 'position', [4 0 8 8]);
subplot(2,1,1)
plot(data_range(1):step:data_range(2),error)
title('Error');

subplot(2,1,2)
plot(X,NN_y)
hold on
plot(X,d)

title('Y and D')
legend('NN output','Validation Data');

