cd('C:/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha');

clearvars
close all
%clc

addpath('C:/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/DeepMIMO Dataset');
addpath('C:/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/MAT functions');
addpath('C:/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/RayTracing Scenarios/O1_28');

seed=0;
rng(seed, "twister") % Added for code replicability
% rng("default") initializes the MATLAB random number generator
% using the default algorithm and seed. The factory default is 
% the Mersenne Twister generator with seed 0.

%% Description:
%
% This is the main code for generating Figure 10 in the original article
% mentioned below.
%
% version 1.0 (Last edited: 2019-05-10)
%
% The definitions and equations used in this code refer (mostly) to the 
% following publication:
%
% Abdelrahman Taha, Muhammad Alrabeiah, and Ahmed Alkhateeb, "Enabling 
% Large Intelligent Surfaces with Compressive Sensing and Deep Learning," 
% arXiv e-prints, p. arXiv:1904.10136, Apr 2019. 
% [Online]. Available: https://arxiv.org/abs/1904.10136
%
% The DeepMIMO dataset is adopted.  
% [Online]. Available: http://deepmimo.net/
%
% License: This code is licensed under a Creative Commons 
% Attribution-NonCommercial-ShareAlike 4.0 International License. 
% [Online]. Available: https://creativecommons.org/licenses/by-nc-sa/4.0/ 
% If you in any way use this code for research that results in 
% publications, please cite our original article mentioned above.

%% System Model parameters

kbeams=1;   %select the top kbeams, get their feedback and find the max actual achievable rate 
Pt=5; % transmit power (dB)
L =1; % number of channel paths
% Note: The axes of the antennas match the axes of the ray-tracing scenario
%My_ar=[32 64]; % number of LIS reflecting elements across the y axis (32x32 blue curve, 64x64 red curve)
%Mz_ar=[32 64]; % number of LIS reflecting elements across the z axis
My_ar=[32]; % Semplificazione Luca
Mz_ar=[32]; % Semplificazione Luca
M_bar=8; % number of active elements
K_DL=64; % number of subcarriers as input to the Deep Learning model (to reduce the neural network complexity)

Ur_rows = [1000 1200]; % original
%Ur_rows = [1000 1300]; % paper

%Training_Size=[2  1e4*(1:.4:3)]; % Training Dataset Size vector (x-axis of Fig 12)
Training_Size=[1e4]; % Semplificazione Luca

% Preallocation of output variables (y-axis of Fig 12 for both blue and red curves)
Rate_DLt=zeros(numel(My_ar),numel(Training_Size));  % numel = number of elements
Rate_OPTt=zeros(numel(My_ar),numel(Training_Size));

%% Figure Data Generation 

for rr = 1:1:numel(My_ar)
    [Rate_DL,Rate_OPT]=Main_fn(seed,L,My_ar(rr),Mz_ar(rr),M_bar,K_DL,Pt,kbeams,Ur_rows,Training_Size);
    Rate_DLt(rr,:)=Rate_DL; Rate_OPTt(rr,:)=Rate_OPT;

    sfile_DeepMIMO=strcat('./Fig12_data', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar), num2str(Mz_ar), '_Mbar', num2str(M_bar), '_', num2str(numel(Training_Size)), '.mat');
    save(sfile_DeepMIMO, 'L', 'My_ar', 'Mz_ar', 'M_bar', 'Training_Size', 'K_DL', 'Rate_DLt', 'Rate_OPTt');
end

%save Fig12_data.mat L My_ar Mz_ar M_bar Training_Size K_DL Rate_DLt Rate_OPTt

%% Figure Plot

%------------- Figure Input Variables ---------------------------%
% M; My_ar; Mz_ar; M_bar; 
% Training_Size; Rate_DLt; Rate_OPTt;

%------------------ Fixed Parameters ----------------------------%
% Full Regression 
% L = min = 1
% K = 512, K_DL = max = 64
% M_bar = 8
% random distribution of active elements

Colour = 'brgmcky';

f12 = figure('Name', 'Figure12', 'units','pixels');
hold on; grid on; box on;
title(['Data Scaling Curve with ' num2str(M_bar) ' active elements'],'fontsize',12)
xlabel('Deep Learning Training Dataset Size (Thousands of Samples)','fontsize',14)
ylabel('Achievable Rate (bps/Hz)','fontsize',14)
set(gca,'FontSize',13)
if ishandle(f12)
    set(0, 'CurrentFigure', f12)
    hold on; grid on;
    for rr=1:1:numel(My_ar)
        plot((Training_Size*1e-3),Rate_OPTt(rr,:),[Colour(rr) '*--'],'markersize',8,'linewidth',2, 'DisplayName',['Genie-Aided Reflection Beamforming, M = ' num2str(My_ar(rr)) '*' num2str(Mz_ar(rr))])
        plot((Training_Size*1e-3),Rate_DLt(rr,:),[Colour(rr) 's-'],'markersize',8,'linewidth',2, 'DisplayName', ['DL Reflection Beamforming, M = ' num2str(My_ar(rr)) '*' num2str(Mz_ar(rr))])
    end
    %legend('Location','SouthEast')
    legend('Location','NorthWest')
    legend show
    ylim([0 3]);
end
drawnow
hold off

sfile_DeepMIMO=strcat('./Fig12', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar), num2str(Mz_ar), '_Mbar', num2str(M_bar), '_', num2str(numel(Training_Size)), '.png');
saveas(f12, sfile_DeepMIMO); % Save the figure to a file 
close(f12); % Close the figure drawnow hold off

disp('End of Fig12_generator.m');
