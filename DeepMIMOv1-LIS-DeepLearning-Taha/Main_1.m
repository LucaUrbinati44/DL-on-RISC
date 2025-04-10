clearvars
close all
%clc

base_folder = 'C:/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/';
output_folder = [base_folder, 'Output Matlab/'];
output_folder_py = [base_folder, 'Output Python/'];

global seed DeepMIMO_dataset_folder DL_dataset_folder network_folder network_folder_py figure_folder figure_folder_py;

seed=0;
rng(seed, "twister") % Added for code replicability
% rng("default") initializes the MATLAB random number generator
% using the default algorithm and seed. The factory default is 
% the Mersenne Twister generator with seed 0.

DeepMIMO_dataset_folder = [output_folder, 'DeepMIMO Dataset/'];
DL_dataset_folder = [output_folder, 'DL Dataset/'];
network_folder = [output_folder, 'Neural Network/'];
figure_folder = [output_folder, 'Figures/'];

network_folder_py = [output_folder_py, 'Neural Network/'];
figure_folder_py = [output_folder_py, 'Figures/'];

folders = {DeepMIMO_dataset_folder DL_dataset_folder network_folder network_folder_py figure_folder figure_folder_py};
for i = 1:length(folders)
    if ~exist(folders{i}, 'dir') % Controlla se la cartella esiste
        mkdir(folders{i}); % Crea la cartella se non esiste
        disp(['Cartella creata: ', folders{i}]);
    else
        disp(['La cartella esiste già: ', folders{i}]);
    end
end
        
addpath(DeepMIMO_dataset_folder);
addpath(DL_dataset_folder);
addpath(figure_folder);
addpath(figure_folder_py);
addpath('C:/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/MAT functions');
addpath('C:/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/RayTracing Scenarios/O1_28');

cd(base_folder);

%%

% Luca variables to control the flow
global load_H_files load_Delta_H_max load_DL_dataset load_Rates training save_mat_files load_mat;

load_Delta_H_max = 1; % load output from DeepMIMO_data_generator_2.m
load_H_files     = 1; % load output from DeepMIMO_data_generator_2.m
load_DL_dataset  = 1; % load output from DL_data_generator_3.m
load_Rates       = 1; % load output from DL_training_4.m
training         = 0; % 1 for training the network, 0 from loaoding it from file
save_mat_files   = 0;

load_mat         = 0; % load mat files generated in Python

plot_fig12 = 1;
plot_fig7 = 0;

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
My_ar=[32 64]; % number of LIS reflecting elements across the y axis (32x32 blue curve, 64x64 red curve)
Mz_ar=[32 64]; % number of LIS reflecting elements across the z axis
%My_ar=[32]; % Semplificazione Luca
%Mz_ar=[32]; % Semplificazione Luca
%My_ar=[64]; % Semplificazione Luca
%Mz_ar=[64]; % Semplificazione Luca
disp('RIS sizes: ');
disp(My_ar);
disp(Mz_ar);

M_bar=8; % number of active elements

D_Lambda = 0.5; % Antenna spacing relative to the wavelength
BW = 100e6; % Bandwidth
K = 512; % number of subcarriers
K_DL=64; % number of subcarriers as input to the Deep Learning model (to reduce the neural network complexity)

Ur_rows = [1000 1200]; % original
%Ur_rows = [1000 1300]; % paper

Training_Size=[2  1e4*(1:.4:3)]; % Training Dataset Size vector (x-axis of Fig 12)
%Training_Size=[1e4*3]; % Semplificazione Luca
%Training_Size=[26000 30000];

% Preallocation of output variables (y-axis of Fig 12 for both blue and red curves)
Rate_DLt=zeros(numel(My_ar),numel(Training_Size));  % numel = number of elements
Rate_OPTt=zeros(numel(My_ar),numel(Training_Size));

for rr = 1:1:numel(My_ar)

    % Note: The axes of the antennas match the axes of the ray-tracing scenario
    Mx = 1;  % number of LIS reflecting elements across the x axis
    My=My_ar(rr);
    Mz=Mz_ar(rr);

    %% DeepMIMO Dataset Generation

    [Ur_rows_grid]=DeepMIMO_data_generator_2(Mx,My,Mz,D_Lambda,BW,K,K_DL,L,Ut_row,Ut_element,Ur_rows,params);

    %% Deep Learning Dataset Generation = Genie-Aided Reflection Beamforming

    [RandP_all,Validation_Ind]=DL_data_generator_3(Mx,My,Mz,M_bar,D_Lambda,BW,K,K_DL,Pt,Ur_rows,Ut_element,Ur_rows_grid);

    %% DL Beamforming

    for dd=1:1:numel(Training_Size)
        [Rate_OPT,Rate_DL]=DL_training_4(Mx,My,Mz,M_bar,Ur_rows,kbeams,Training_Size(dd),RandP_all,Validation_Ind);
        %[Rate_OPT,Rate_DL]=DL_predict_5(Mx,My,Mz,M_bar,Ur_rows,kbeams,Training_Size(dd),RandP_all,Validation_Ind);
        Rate_OPTt(rr,dd)=Rate_OPT;
        Rate_DLt(rr,dd)=Rate_DL;
    end

    %keyboard;

end

%% Fig 12
if plot_fig12 == 1
    Fig12_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,Training_Size,Rate_OPTt,Rate_DLt);
end

%% Fig 7 (Luca)

if plot_fig7 == 1
    Fig7_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams);
end

%% End of script

%disp('--> End of script. If you continue, you will lose access to all variables');

%keyboard;