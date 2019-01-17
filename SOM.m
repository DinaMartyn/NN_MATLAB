%% SOM
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Script is for the Exercise for Soft Control and should help understand the 
% Self Organizing Map. Some useful Functions are given as well as the Outline of the Script.
% Please fill in all blanks as indicated by comments.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
%data sample: colors in RGB
%task: group 23 colors in 9 groups 
% use SOM with 3 inputs and 9 outputs.
training_colors=[
    0 0.9 0.2; 0.2 0.5 0.7; 1 1 0.3; 0.6 0.2 0.1; 1 0 0;
    0 0.1 0.7; 1 1 1; 0.2 0.6 0.2; 0 0 0; 0 0 1;
    0 0 0.5; 0.1 0.5 1; 0.3 0.4 0.6; 0.6 0.5 1; 0 1 0;
    0.8 0.1 0.2; 0 1 1; 1 0 1; 1 0.8 0.2; 1 0.8 0.9;
    0.3 0.5 0.3; 0.5 0.7 0.2; 0.6 0.6 0.6;
 ];
%starting weights so that every trial starts with the same starting point.
starting_weights=  [
    0.03    0.15    0.03;
    1.0	    0.80	0.10;
    0.70	0.930	0.80;
    0.15	0.01	0.70;
    0.15	0.65	0.75;
    0.70	0.68	0.62;
    0.38	0.25	0.82;
    0.07	0.840	0.04;
    0.41	0.30	0.41;
];


number_of_neurons=9; %must be a squared number and musst be the same number as the starting weights: each neuron is associatied with one of the startin weights.
%if the number is changed the starting weights also have to be changed.
%starting weights generated with rand(number_of_neurons,3);

weights = starting_weights;
 
%learning rate: specify a learning rate which decreases over time (Script
% 07, slide 15)
eta=@(time) 0.5*exp(-time/1000);

%neighborhood function: specify a neighborhood function dependend on the
%distance to the winner and the time
h =@(winner,neuron,time) exp(-abs(winner-neuron).^2/(2*4*exp(-time./100)));

% added for visualisation
a = figure;
a.Units = 'normalized';
a.Position = [0 0 0.45 0.85];
hold on

for time=1:500
% select randomly a trainign sample from the data set:
training_sample=training_colors(randi(size(training_colors,1)),:);

%find the winning neuron: calculate the distance to the training sample and
%select the shortest distance
distance= zeros(size(starting_weights,1),1);
for i = 1:size(starting_weights,1)
    distance(i) = norm(training_sample-weights(i,:));
    [~,winner] = min(distance);
end
%update the neurons:
for i=1:number_of_neurons
    weights(i,:)=weights(i,:)+eta(time)*h(winner,i,time)*(training_sample-weights(i,:));
end
%%
belong = zeros(size(training_colors,1),1);
for j = 1:size(training_colors,1)
    for i = 1:size(starting_weights,1)
        distance(i) = norm(training_colors(j,:) - weights(i,:));
        [~,belong(j)] = min(distance);
    end
end
draw_sampl = zeros(size(training_colors,1),4);
k = 1;
for i = 1:size(starting_weights,1)
    current = find(belong==i);
    if isempty(current)
    else
        draw_sampl(k:k+size(current,1)-1,1) = i*ones(size(current));
        draw_sampl(k:k+size(current,1)-1,2:end) = training_colors(current,:);
        k=k+size(current,1);
    end
end
k = 1;
for j = 1:size(draw_sampl,1)
    x = [j j-1 j-1 j];
    y = [2 2 1 1];
    fill(x,y,draw_sampl(j,2:4))
    curr_w = draw_sampl(j,1);
    curr_w_set = find(draw_sampl(:,1)==curr_w);
    if size(curr_w_set,1) > 1
        if k == 1
            fill([j+size(curr_w_set,1)-1 j-1 j-1 j+size(curr_w_set,1)-1],y-1,weights(curr_w,:))
            k = 0;
        end
        if j<23 && draw_sampl(j,1) ~= draw_sampl(j+1,1)
            k = 1;
        end
    else
        fill(x,y-1,weights(curr_w,:))
    end
end
pause(0.01);
end

