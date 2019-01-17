%% RBF: 1 input, 1 output, 1 hidden layer, 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Script is for the Exercise for Soft Control and should help 
% understand the Radial Basis Function Network. 
% Some useful Functions are given as well as the Outline of the Script.
% Please fill in all blanks as indicated by comments.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clear;
close all;

% Data with or without  noise
load RBF_Data.mat
% load RBF_Data_with_noise.mat
% change Y to the function you like
Y = X;
training_points = [X,Y];

% number of hidden neurons
K = 5;   
% initial centers of the activation functions are equally distributed
My = X(1):(X(end)-X(1))/(K-1):X(end); 

% The activation function is a radial basis function
activation_func = @(x,my,sigma) (exp(-(x-my).^2/2/sigma^2)); %Gauss function
% allowed range to training data 
range = -1.5:.01:1.5;
% adjust the given range to the X values in training data
% so that range(X_index) is almost equal X
X_index=zeros(length(X),1);
for i=1:length(X)
  [~,X_index(i)] = min(abs(range-X(i,1))); 
end
% define the value of the radial basis function in each neuron before the training
hiddenlayer_visual_start=zeros(K,length(range));
for i=1:K
    hiddenlayer_visual_start(i,:)=activation_func(range,My(i),0.15);
end
% build F matrix before the trainig
F=hiddenlayer_visual_start(:,X_index)';
% define the weighting factors by applying the least square approach
W=(F'*F)^-1*F'*Y;
% define the value of the radial basis function in each neuron ater the training
hiddenlayer_visual_end=zeros(K,length(range));
for i=1:K
    hiddenlayer_visual_end(i,:)=W(i)*activation_func(range,My(i),0.15);
end
result=sum(hiddenlayer_visual_end,1);
result_start=sum(hiddenlayer_visual_start,1);

figure
hold on
subplot(1,2,1)
hold on;
title('Before training');
xlabel('Input Vector X');
ylabel('Target Vector Y');

 
plot(X,Y,'x','LineWidth',2);
plot(range, result_start,'LineWidth',2);
plot(range,hiddenlayer_visual_start,':')

subplot(1,2,2)
hold on;
title('After training');
xlabel('Input Vector X');
ylabel('Target Vector Y')

 
plot(X,Y,'x','LineWidth',2);
plot(range, result,'LineWidth',2);
% plot(X_val,Y_val,'*','LineWidth',2);
plot(range,hiddenlayer_visual_end,':')
% legend('training Vector','Network ouput','Verification','location','Best');
legend('training Vector','Network ouput','location','Best');