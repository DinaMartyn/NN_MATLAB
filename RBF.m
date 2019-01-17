%% RBF
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Script is for the Exercise for Soft Control and should help understand the 
% Radial Basis Function Network. Some useful Functions are given as well as the Outline of the Script.
% Please fill in all blanks as indicated by comments.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
close all;
%% training data
d=[-.96 -.57 -.072  .37  .64  .66  .46 ...
      .13 -.20 -.43 -.50 -.39 -.16  .09 ...
      .30  .39  .34 .18 -.031 -.21 -.32]';
X = [-1:.1:1]';
training_points = [X d];


%% please fill in the blanks:

 %specify number of neurons  for example 5. Make it a variable to be able to see how the number of neurons affects the result 
 N=5;

 %specify activation function, for example the gaussian function-> myFunc = @(x,y,z) (x+y+z) directly defines a function
 %the activation function is dependend on the sampel as well as the center
 %and the width of the function. use these free paramter to tune your
 %result. 
 %please insert a usable activation function
 activation =@(x,my,sigma) exp(-abs(x-my).^2/(2*sigma^2)) ;
 sigma=0.2;
 %specify centers of the activations functions
 center=-1:2/(N-1):1;
 %define Y and F
 %  Y: target vector, with all points of d
 Y=d;
 %  F: Matrix with the activation function for each data point an input x;
 %  This is easily done in a loop over all neurons and over all data
 %  samples
 for i=1:N
     for k=1:length(X)
     F(k,i)=activation(X(k), center(i), sigma);
     end
 end
 %calculate W
 W=zeros(5,1);
 W=(F'*F)^-1 *F'*Y;
 %plot your results
 figure
 axis([-1,1,-2,2]);
 hold on;
 plot(X,d,'r*'); %data points
result=0;
 for i=1:N
     %original activation funtcions:
     plot(X, activation(X,center(i),sigma),':');
     %weighted activation functions:
     plot(X, W(i)*activation(X,center(i),sigma),'g--');
     result=result+ W(i)*activation(X,center(i),sigma);
 end
 plot(X,result,'k');