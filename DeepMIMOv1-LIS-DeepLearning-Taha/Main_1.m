cd();

clearvars
close all
%clc

base_folder = 'C:/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha';
output_folder = [base_folder, '/Output/'];
dataset_folder = [output_folder, 'DeepMIMO Dataset/'];
figure_folder = [output_folder, 'Figures/'];

addpath([output_folder, 'DeepMIMO Dataset New']);
addpath('C:/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/MAT functions');
addpath('C:/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/RayTracing Scenarios/O1_28');


seed=0;
rng(seed, "twister") % Added for code replicability
% rng("default") initializes the MATLAB random number generator
% using the default algorithm and seed. The factory default is 
% the Mersenne Twister generator with seed 0.

%%

% Luca variables to control the flow
global load_mat_files;
load_mat_files = 1;
global load_Delta_H_max;
load_Delta_H_max = 1;
global load_DL_dataset;
load_DL_dataset = 1;
global load_Rates;
load_Rates = 1;
global save_mat_files;
save_mat_files = 0;

plot_fig12 = 1;
plot_fig7 = 1;

%% System Model parameters
disp('---> System Model Parameters');

params.scenario='O1_28'; % DeepMIMO Dataset scenario: http://deepmimo.net/

Ut_row = 850; % user Ut row number
Ut_element = 90; % user Ut position from the row chosen above

% we select BS 3 in the 'O1' scenario to be the LIS
params.active_BS=3; % active basestation(/s) in the chosen scenario

kbeams=1;   %select the top kbeams, get their feedback and find the max actual achievable rate 
Pt=5; % transmit power (dB)
L =1; % number of channel paths

% Note: The axes of the antennas match the axes of the ray-tracing scenario
%My_ar=[32 64]; % number of LIS reflecting elements across the y axis (32x32 blue curve, 64x64 red curve)
%Mz_ar=[32 64]; % number of LIS reflecting elements across the z axis
My_ar=[32]; % Semplificazione Luca
Mz_ar=[32]; % Semplificazione Luca

% Note: The axes of the antennas match the axes of the ray-tracing scenario
Mx = 1;  % number of LIS reflecting elements across the x axis
M = Mx.*My.*Mz; % Total number of LIS reflecting elements 

M_bar=8; % number of active elements

D_Lambda = 0.5; % Antenna spacing relative to the wavelength
BW = 100e6; % Bandwidth
K = 512; % number of subcarriers
K_DL=64; % number of subcarriers as input to the Deep Learning model (to reduce the neural network complexity)

Ur_rows = [1000 1200]; % original
%Ur_rows = [1000 1300]; % paper

%Training_Size=[2  1e4*(1:.4:3)]; % Training Dataset Size vector (x-axis of Fig 12)
Training_Size=[1e4*3]; % Semplificazione Luca

% Preallocation of output variables (y-axis of Fig 12 for both blue and red curves)
Rate_DLt=zeros(numel(My_ar),numel(Training_Size));  % numel = number of elements
Rate_OPTt=zeros(numel(My_ar),numel(Training_Size));

%% DeepMIMO Dataset Generation

[Ur_rows_grid]=DeepMIMO_data_generator_2(output_folder,dataset_folder,seed,Mx,My,Mz,D_Lambda,BW,K,K_DL,L);

%% Deep Learning Dataset Generation = Genie-Aided Reflection Beamforming

DL_data_generator_3(output_folder,dataset_folder,seed,Mx,My,Mz,D_Lambda,BW,K,K_DL,Ur_rows,M_bar,Ur_rows_grid);

%% DL Beamforming

for dd=1:1:numel(Training_Size)
    [Rate_OPT,Rate_DL] = DL_training_4(Training_Size);
    Rate_OPTt(dd,:)=Rate_OPT;
    Rate_DLt(dd,:)=Rate_DL;
end

%% Fig 12

if plot_fig12 == 1
    Fig12_plot(output_folder,seed,M_bar,Ur_rows,Training_Size,Rate_OPTt,Rate_DLt);
end

%% Fig 7 (Luca)

if plot_fig7 == 1

    for i=1:1:2
        correct_fig7 = i-1;
        Fig7_plot(filename_DL_input_reshaped, ...
                    filename_DL_output_reshaped, ...
                    filename_trainedNet, ...
                    filename_YPredictedFig7, ...
                    Ur_rows, ...
                    kbeams, ...
                    output_folder, ...
                    save_mat_files, ...
                    correct_fig7);
    end
end

%% End of script

disp('--> End of script. If you continue, you will lose access to all variables');

%keyboard;