clearvars
close all
%clc

base_folder = 'C:/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/';
output_folder = [base_folder, 'Output Matlab/'];
output_folder_py = [base_folder, 'Output_Python/'];

global seed DeepMIMO_dataset_folder end_folder end_folder_M_bar DL_dataset_folder network_folder network_folder_py figure_folder figure_folder_py network_folder_out_RateDLpy network_folder_out_RateDLpy_TFLite;

seed=0;
rng(seed, "twister") % Added for code replicability
% rng("default") initializes the MATLAB random number generator
% using the default algorithm and seed. The factory default is 
% the Mersenne Twister generator with seed 0.

DeepMIMO_dataset_folder = [output_folder, 'DeepMIMO Dataset/'];
DL_dataset_folder = [output_folder, 'DL Dataset/'];
network_folder = [output_folder, 'Neural Network/'];
figure_folder = [output_folder, 'Figures/'];

network_folder_py = [output_folder_py, 'Neural_Network/'];
figure_folder_py = [output_folder_py, 'Figures/'];
network_folder_out_RateDLpy = [network_folder_py, 'RateDLpy/'];
network_folder_out_RateDLpy_TFLite = [network_folder_py, 'RateDLpy_TFLite/'];

folders = {DeepMIMO_dataset_folder DL_dataset_folder network_folder network_folder_py figure_folder figure_folder_py network_folder_out_RateDLpy network_folder_out_RateDLpy_TFLite};
for i = 1:length(folders)
    if ~exist(folders{i}, 'dir') % Controlla se la cartella esiste
        mkdir(folders{i}); % Crea la cartella se non esiste
        disp(['Cartella creata: ', folders{i}]);
    %else
        %disp(['La cartella esiste già: ', folders{i}]);
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
global load_H_files load_Delta_H_max load_DL_dataset load_Rates training save_mat_files load_mat_py;

% all 1s in production
load_Delta_H_max = 1; % load output from DeepMIMO_data_generator_2.m
load_H_files     = 1; % load output from DeepMIMO_data_generator_2.m
load_DL_dataset  = 0; % load output from DL_data_generator_3.m
load_Rates       = 0; % load output from DL_training_4.m
training         = 0; % 1 for training the network, 0 from loaoding it from file
save_mat_files   = 1;

load_mat_py      = 4; 
% 4: load py-generated test tflite files (production)
% 3: load py-generated test files
% 2: load py-generated files
% 1: load mat-python-mat files
% 0: load mat-generated files

%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot_fig12 = 0;
epochs_fig12 = 60;
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot_fig7 = 0;
plot_figC = 0;
plot_index     = 0; % con plot_index=1 si può usare solo plot_test_only=1 (per ora perchè bisogna plottare gli indici il cui rate è superiore alla soglia (non so se è di interesse))
plot_rate      = 1;
plot_test_only = 0;
plot_threshold = 0;
threshold      = 3; % [bps/Hz], only if plot_threshold == 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot_figD = 0; % Richiede di impostare My_ar=[32 64]; Mz_ar=[32 64];
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Settings condivise tra i vari plot
plot_mode = 2; % 1: matlab, 2: python
Training_Size_number = 6;
%Training_Size_number = 11;
%                1    2     3     4     5     6      7      8      9      10     11
Training_Size = [2, 2000, 4000, 6000, 8000, 10000, 14000, 18000, 22000, 26000, 30000];
%max_epochs_load = 20;
%max_epochs_load = 40;
max_epochs_load = 60;
%max_epochs_load = 80; % ATTENZIONE funziona solo con [64, 64] per ora
%max_epochs_load = 100; % ATTENZIONE funziona solo con [64, 64] per ora

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% System Model parameters
disp('---> System Model Parameters');

params.scenario='O1_28'; % DeepMIMO Dataset scenario: http://deepmimo.net/

Ut_row = 850; % user Ut row number
Ut_element = 90; % user Ut position from the row chosen above

% we select BS 3 in the 'O1' scenario to be the LIS
params.active_BS=3; % active basestation(/s) in the chosen scenario

kbeams=1;   %select the top kbeams, get their feedback and find the max actual achievable rate 
Pt=5; % transmit power (dB)
L=1; % number of channel paths

% TO CHANGE FOR DESIGN-SPACE EXPLORATION
M_bar=8; % number of active elements

D_Lambda = 0.5; % Antenna spacing relative to the wavelength
BW = 100e6; % Bandwidth
K = 512; % number of subcarriers
K_DL=64; % number of subcarriers as input to the Deep Learning model (to reduce the neural network complexity)

Ur_rows = [1000 1200]; % original
%Ur_rows = [1000 1300]; % paper

%Training_Size=[2  1e4*(1:.4:3)]; % Training Dataset Size vector (x-axis of Fig 12)
%Training_Size=[1e4*3]; % Semplificazione Luca
%Training_Size=[10000 30000];
%Training_Size = [2, 2000, 4000, 6000, 8000, 10000, 14000, 18000, 22000, 26000, 30000];

Validation_Size = 6200; % Validation dataset Size
Test_Size = 3100; % Test dataset Size

% fig12
Rate_OPTt=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_mat=zeros(numel(My_ar),numel(Training_Size));  % numel = number of elements

Rate_DLt_py_valOld_20=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_valOld_40=zeros(numel(My_ar),numel(Training_Size));

Rate_DLt_py_val_20=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_val_40=zeros(numel(My_ar),numel(Training_Size));
%Rate_DLt_py_val_60=zeros(numel(My_ar),numel(Training_Size));
%Rate_DLt_py_val_80=zeros(numel(My_ar),numel(Training_Size));

Rate_DLt_py_test_20=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_test_40=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_test_60=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_test_80=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_test_100=zeros(numel(My_ar),numel(Training_Size));

Rate_DLt_py_test_tflite_20=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_test_tflite_40=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_test_tflite_60=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_test_tflite_80=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_test_tflite_100=zeros(numel(My_ar),numel(Training_Size));

% figD
MaxR_OPTt = single(zeros(numel(My_ar),numel(Training_Size),Validation_Size));
MaxR_DLt_mat = single(zeros(numel(My_ar),numel(Training_Size),Validation_Size));

%MaxR_OPTt_py_val = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));

%MaxR_DLt_py_val_20 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
%MaxR_DLt_py_val_40 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
%MaxR_DLt_py_val_60 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
%MaxR_DLt_py_val_80 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
%MaxR_DLt_py_val_100 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));

MaxR_OPTt_py_test = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_OPTt_py_test_20 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_OPTt_py_test_40 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_OPTt_py_test_60 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_OPTt_py_test_80 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_OPTt_py_test_100 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));

MaxR_DLt_py_test_20 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_DLt_py_test_40 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_DLt_py_test_60 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_DLt_py_test_80 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_DLt_py_test_100 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));

MaxR_DLt_py_test_tflite_20 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_DLt_py_test_tflite_40 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_DLt_py_test_tflite_60 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_DLt_py_test_tflite_80 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_DLt_py_test_tflite_100 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));

for rr = 1:1:numel(My_ar)

    % Note: The axes of the antennas match the axes of the ray-tracing scenario
    Mx = 1;  % number of LIS reflecting elements across the x axis
    My=My_ar(rr);
    Mz=Mz_ar(rr);
    
    end_folder = strcat('_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My), ' ', ''), strrep(num2str(Mz), ' ', ''));
    end_folder_M_bar = strcat(end_folder, '_Mbar', num2str(M_bar));

    %% DeepMIMO Dataset Generation

    [Ur_rows_grid]=DeepMIMO_data_generator_2(Mx,My,Mz,D_Lambda,BW,K,K_DL,L,Ut_row,Ut_element,Ur_rows,params);

    %% Deep Learning Dataset Generation = Genie-Aided Reflection Beamforming

    [RandP_all,Validation_Ind]=DL_data_generator_3(Mx,My,Mz,M_bar,D_Lambda,BW,K,K_DL,Pt,Ur_rows,Ut_element,Ur_rows_grid,Validation_Size);

    %Test_Ind = RandP_all2[-Test_Size:] % python
    Test_Ind = RandP_all(end-Test_Size+1:end);
    %Validation_Ind = RandP_all2[-(Validation_Size+Test_Size):-Test_Size] % python
    %Validation_Ind_6200 = RandP_all(end-(Test_Size+Test_Size)+1:end);

    continue
    
    %% DL Beamforming

    for dd=1:1:numel(Training_Size)

        Training_Size_dd = Training_Size(dd);

        if Training_Size_dd >= 10000 || Training_Size_dd == 2
            [Rate_OPT,Rate_DL,MaxR_OPT,MaxR_DL]=DL_training_4(Mx,My,Mz,M_bar,Ur_rows,kbeams,Training_Size(dd),RandP_all,Validation_Ind);
            %[Rate_OPT,Rate_DL]=DL_predict_5(Mx,My,Mz,M_bar,Ur_rows,kbeams,Training_Size(dd),RandP_all,Validation_Ind);
        else
            Rate_OPT = NaN;
            Rate_DL = NaN;
            MaxR_OPT = NaN;
            MaxR_DL = NaN;
        end
            
        if plot_figD == 1 || plot_figC == 1

            MaxR_OPTt(rr,dd,:) = MaxR_OPT; % 6200 val
            MaxR_DLt_mat(rr,dd,:) = MaxR_DL; % 6200 val

            %hasNaN = any(isnan(MaxR_DL));
            %if hasNaN
            %    disp('MaxR_DL contiene NaN.');
            %else
            %    disp('MaxR_DL NON contiene NaN.');
            %end

            % Load MaxR_DLt_py from Python

            %plot_types = {'MaxR_DLt_py_valOld' 'MaxR_OPTt_py_val' 'MaxR_DLt_py_val' 'MaxR_OPTt_py_test' 'MaxR_DLt_py_test'};
            %plot_types = {'MaxR_OPT_py_val' 'MaxR_DL_py_val' 'MaxR_OPT_py_test' 'MaxR_DL_py_test'};
            plot_types = {'MaxR_DL_py_val' 'MaxR_OPT_py_test' 'MaxR_DL_py_test' 'MaxR_DL_py_test_tflite'};

            for epochs = 20:20:max_epochs_load
            
                for i = 1:length(plot_types)

                    % TODO controllare che gli indici di MaxR_DL_py siano nello stesso ordine di quelli di MaxR_DL (matlab).
                    end_folder_Training_Size_dd_max_epochs = strcat(end_folder, '_', num2str(Training_Size_dd), '_', num2str(epochs));
                    
                    try
                        if strcmp(plot_types{i}, 'MaxR_OPT_py_val') || strcmp(plot_types{i}, 'MaxR_OPT_py_test')
                            filename_MaxR_OPT_py = strcat(network_folder_out_RateDLpy, plot_types{i}, end_folder_Training_Size_dd_max_epochs, '.mat');
                            MaxR_OPT_py = h5read(filename_MaxR_OPT_py, '/MaxR_OPT_py'); % 3100 test
                        elseif strcmp(plot_types{i}, 'MaxR_DL_py_val') || strcmp(plot_types{i}, 'MaxR_DL_py_test')
                            filename_MaxR_DL_py = strcat(network_folder_out_RateDLpy, plot_types{i}, end_folder_Training_Size_dd_max_epochs, '.mat');
                            MaxR_DL_py = h5read(filename_MaxR_DL_py, '/MaxR_DL_py'); % 3100 test
                        elseif strcmp(plot_types{i}, 'MaxR_DL_py_test_tflite')
                            filename_MaxR_DL_py = strcat(network_folder_out_RateDLpy_TFLite, plot_types{i}, end_folder_Training_Size_dd_max_epochs, '.mat');
                            MaxR_DL_py = h5read(filename_MaxR_DL_py, '/MaxR_DL_py'); % 3100 test
                        end
                    catch exception
                        MaxR_OPT_py = NaN;
                        MaxR_DL_py = NaN;
                        disp('exception')
                        disp(filename_MaxR_DL_py)
                    end

                    if strcmp(plot_types{i}, 'MaxR_OPT_py_val') % OPT val set
                        %MaxR_OPTt_py_val(rr,dd,:) = MaxR_OPT_py;

                    elseif strcmp(plot_types{i}, 'MaxR_DL_py_val') % Val set
                        %if epochs == 20
                        %    MaxR_DLt_py_val_20(rr,dd,:) = MaxR_DL_py;
                        %elseif epochs == 40
                        %    MaxR_DLt_py_val_40(rr,dd,:) = MaxR_DL_py;
                        %elseif epochs == 60
                        %    MaxR_DLt_py_val_60(rr,dd,:) = MaxR_DL_py;
                        %elseif epochs == 80
                        %    MaxR_DLt_py_val_80(rr,dd,:) = MaxR_DL_py;
                        %elseif epochs == 100
                        %    MaxR_DLt_py_val_100(rr,dd,:) = MaxR_DL_py;
                        %end
                    elseif strcmp(plot_types{i}, 'MaxR_OPT_py_test') % OPT Test set
                        MaxR_OPTt_py_test(rr,dd,:) = MaxR_OPT_py;
                        if epochs == 20
                            MaxR_OPTt_py_test_20(rr,dd,:) = MaxR_OPT_py;
                        elseif epochs == 40
                            MaxR_OPTt_py_test_40(rr,dd,:) = MaxR_OPT_py;
                        elseif epochs == 60
                            MaxR_OPTt_py_test_60(rr,dd,:) = MaxR_OPT_py;
                        elseif epochs == 80
                            MaxR_OPTt_py_test_80(rr,dd,:) = MaxR_OPT_py;
                        elseif epochs == 100
                            MaxR_OPTt_py_test_100(rr,dd,:) = MaxR_OPT_py;
                        end
                    elseif strcmp(plot_types{i}, 'MaxR_DL_py_test') % Test set
                        if epochs == 20
                            MaxR_DLt_py_test_20(rr,dd,:) = MaxR_DL_py;
                            %hasNaN = any(isnan(MaxR_DL_py));
                            %if hasNaN
                            %    disp('MaxR_DL_py contiene NaN.');
                            %else
                            %    disp('MaxR_DL_py NON contiene NaN.');
                            %end
                        elseif epochs == 40
                            MaxR_DLt_py_test_40(rr,dd,:) = MaxR_DL_py;
                        elseif epochs == 60
                            MaxR_DLt_py_test_60(rr,dd,:) = MaxR_DL_py;
                        elseif epochs == 80
                            MaxR_DLt_py_test_80(rr,dd,:) = MaxR_DL_py;
                        elseif epochs == 100
                            MaxR_DLt_py_test_100(rr,dd,:) = MaxR_DL_py;
                        end
                    elseif strcmp(plot_types{i}, 'MaxR_DL_py_test_tflite') % Test set tflite
                        if epochs == 20
                            MaxR_DLt_py_test_tflite_20(rr,dd,:) = MaxR_DL_py;
                        elseif epochs == 40
                            MaxR_DLt_py_test_tflite_40(rr,dd,:) = MaxR_DL_py;
                        elseif epochs == 60
                            MaxR_DLt_py_test_tflite_60(rr,dd,:) = MaxR_DL_py;
                        elseif epochs == 80
                            MaxR_DLt_py_test_tflite_80(rr,dd,:) = MaxR_DL_py;
                        elseif epochs == 100
                            MaxR_DLt_py_test_tflite_100(rr,dd,:) = MaxR_DL_py;
                        end
                    end
                end
            end
        end

        if plot_fig12 == 1

            Rate_OPTt(rr,dd)=Rate_OPT; % 6200 val
            Rate_DLt_mat(rr,dd)=Rate_DL; % 6200 val

            % Load Rate_DL_py from Python

            plot_types = {'Rate_DL_py_valOld' 'Rate_DL_py_val' 'Rate_DL_py_test' 'Rate_DL_py_test_tflite'};
            
            for epochs = 20:20:epochs_fig12
            
                for i = 1:length(plot_types)
                    
                    end_folder_Training_Size_dd_max_epochs = strcat(end_folder, '_', num2str(Training_Size_dd), '_', num2str(epochs));

                    if strcmp(plot_types{i}, 'Rate_DL_py_test_tflite') % Test set TFLite
                        filename_Rate_DL_py = strcat(network_folder_out_RateDLpy_TFLite, plot_types{i}, end_folder_Training_Size_dd_max_epochs, '.mat');
                    else
                        filename_Rate_DL_py = strcat(network_folder_out_RateDLpy, plot_types{i}, end_folder_Training_Size_dd_max_epochs, '.mat');
                    end
                    
                    try
                        Rate_DL_py = h5read(filename_Rate_DL_py, '/Rate_DL_py');
                    catch exception
                        Rate_DL_py = NaN;
                        disp('exception')
                        disp(filename_Rate_DL_py)
                    end
                    
                    if strcmp(plot_types{i}, 'Rate_DL_py_valOld') % valOld
                        if epochs == 20
                            Rate_DLt_py_valOld_20(rr,dd)= Rate_DL_py;
                        elseif epochs == 40
                            Rate_DLt_py_valOld_40(rr,dd)= Rate_DL_py;
                        end

                    
                    elseif strcmp(plot_types{i}, 'Rate_DL_py_val') % Val set
                        if epochs == 20
                            Rate_DLt_py_val_20(rr,dd)= Rate_DL_py;
                        elseif epochs == 40
                            Rate_DLt_py_val_40(rr,dd)= Rate_DL_py;
                        end
                    
                    elseif strcmp(plot_types{i}, 'Rate_DL_py_test') % Test set
                        if epochs == 20
                            Rate_DLt_py_test_20(rr,dd)= Rate_DL_py;
                        elseif epochs == 40
                            Rate_DLt_py_test_40(rr,dd)= Rate_DL_py;
                        elseif epochs == 60
                            Rate_DLt_py_test_60(rr,dd)= Rate_DL_py;
                        elseif epochs == 80
                            Rate_DLt_py_test_80(rr,dd)= Rate_DL_py;
                        elseif epochs == 100
                            Rate_DLt_py_test_100(rr,dd)= Rate_DL_py;
                        end
                        
                    elseif strcmp(plot_types{i}, 'Rate_DL_py_test_tflite') % Test set TFLite
                        if epochs == 20
                            Rate_DLt_py_test_tflite_20(rr,dd)= Rate_DL_py;
                        elseif epochs == 40
                            Rate_DLt_py_test_tflite_40(rr,dd)= Rate_DL_py;
                        elseif epochs == 60
                            Rate_DLt_py_test_tflite_60(rr,dd)= Rate_DL_py;
                        elseif epochs == 80
                            Rate_DLt_py_test_tflite_80(rr,dd)= Rate_DL_py;
                        elseif epochs == 100
                            Rate_DLt_py_test_tflite_100(rr,dd)= Rate_DL_py;
                        end

                    end

                end
            end
        end
    end
end

%% Fig 12
if plot_fig12 == 1
    Fig12_plot(My_ar,Mz_ar,M_bar,Ur_rows,Training_Size,...
                epochs_fig12, ...
                Rate_OPTt,Rate_DLt_mat, ...
                Rate_DLt_py_valOld_20,Rate_DLt_py_valOld_40, ...
                Rate_DLt_py_val_20,Rate_DLt_py_test_20,Rate_DLt_py_test_tflite_20, ...
                Rate_DLt_py_val_40,Rate_DLt_py_test_40,Rate_DLt_py_test_tflite_40, ...
                Rate_DLt_py_test_60,Rate_DLt_py_test_tflite_60, ...
                Rate_DLt_py_test_80,Rate_DLt_py_test_tflite_80, ...
                Rate_DLt_py_test_100, Rate_DLt_py_test_tflite_100)

               
end

%% Fig 7 (Luca)

%if plot_fig7 == 1
%    Fig7_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams);
%end

%% Fig C (Luca)

if plot_figC == 1
    
    % mat
    %plot_mode = 1;
    %%max_epochs_load = 20; % DO NOT CHANGE, matVal ha solo questo valore di epoche
    %FigC_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,RandP_all,Validation_Ind,Test_Ind,epochs,plot_mode,Training_Size_number,...
    %            plot_index,plot_rate,plot_test_only,plot_threshold,threshold,...
    %            MaxR_OPTt, MaxR_DLt_mat, ...
    %            MaxR_OPTt_py_test, ...
    %            MaxR_DLt_py_test_20, ...
    %            MaxR_DLt_py_test_40, ...
    %            MaxR_DLt_py_test_60, ...
    %            MaxR_DLt_py_test_80, ...
    %            MaxR_DLt_py_test_100);
    
    % py
    %plot_mode = 2;
    %max_epochs_load = 20;
    %max_epochs_load = 100;
    FigC_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,RandP_all,Validation_Ind,Test_Ind,epochs,plot_mode,Training_Size_number,...
                plot_index,plot_rate,plot_test_only,plot_threshold,threshold,...
                MaxR_OPTt, MaxR_DLt_mat, ...
                MaxR_OPTt_py_test, ...
                MaxR_DLt_py_test_20, ...
                MaxR_DLt_py_test_40, ...
                MaxR_DLt_py_test_60, ...
                MaxR_DLt_py_test_80, ...
                MaxR_DLt_py_test_100);

    FigC_plot_tflite(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,RandP_all,Validation_Ind,Test_Ind,epochs,plot_mode,Training_Size_number,...
                plot_index,plot_rate,plot_test_only,plot_threshold,threshold,...
                MaxR_OPTt, MaxR_DLt_mat, ...
                MaxR_OPTt_py_test, ...
                MaxR_DLt_py_test_20, MaxR_DLt_py_test_tflite_20, ...
                MaxR_DLt_py_test_40, MaxR_DLt_py_test_tflite_40, ...
                MaxR_DLt_py_test_60, MaxR_DLt_py_test_tflite_60, ...
                MaxR_DLt_py_test_80, MaxR_DLt_py_test_tflite_80, ...
                MaxR_DLt_py_test_100, MaxR_DLt_py_test_tflite_100);
end

if plot_figD == 1

    FigD_plot(My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,Validation_Ind,Test_Ind,max_epochs_load,plot_mode,Training_Size_number, ...
                MaxR_OPTt,MaxR_DLt_mat, ...
                MaxR_OPTt_py_test, ...
                MaxR_DLt_py_test_20, ...
                MaxR_DLt_py_test_40, ...
                MaxR_DLt_py_test_60, ...
                MaxR_DLt_py_test_80, ...
                MaxR_DLt_py_test_100)

    FigD_plot_tflite(My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,Validation_Ind,Test_Ind,max_epochs_load,plot_mode,Training_Size_number, ...
                MaxR_OPTt,MaxR_DLt_mat, ...
                MaxR_OPTt_py_test, ...
                MaxR_DLt_py_test_20, MaxR_DLt_py_test_tflite_20, ...
                MaxR_DLt_py_test_40, MaxR_DLt_py_test_tflite_40, ...
                MaxR_DLt_py_test_60, MaxR_DLt_py_test_tflite_60, ...
                MaxR_DLt_py_test_80, MaxR_DLt_py_test_tflite_80, ...
                MaxR_DLt_py_test_100, MaxR_DLt_py_test_tflite_100)

    % ATTENZIONE: non è adatto a plottare ris 32 e 64 sullo stesso grafico, plottare uno alla volta.
    M_ar_master_list = [32 64];
    for i=1:1:2
        
        M_ar_master = M_ar_master_list(i);

        if M_ar_master == 32 && max_epochs_load > 60
            continue
        end

        FigD_plot_all(My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,Validation_Ind,Test_Ind,max_epochs_load,plot_mode,Training_Size_number,M_ar_master, ...
                    MaxR_OPTt,MaxR_DLt_mat, ...
                    MaxR_OPTt_py_test, ...
                    MaxR_DLt_py_test_20, ...
                    MaxR_DLt_py_test_40, ...
                    MaxR_DLt_py_test_60, ...
                    MaxR_DLt_py_test_80, ...
                    MaxR_DLt_py_test_100)

        FigD_plot_all_tflite(My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,Validation_Ind,Test_Ind,max_epochs_load,plot_mode,Training_Size_number,M_ar_master, ...
                    MaxR_OPTt,MaxR_DLt_mat, ...
                    MaxR_OPTt_py_test, ...
                    MaxR_DLt_py_test_20,  MaxR_DLt_py_test_tflite_20, ...
                    MaxR_DLt_py_test_40,  MaxR_DLt_py_test_tflite_40, ...
                    MaxR_DLt_py_test_60,  MaxR_DLt_py_test_tflite_60, ...
                    MaxR_DLt_py_test_80,  MaxR_DLt_py_test_tflite_80, ...
                    MaxR_DLt_py_test_100, MaxR_DLt_py_test_tflite_100)
    end

end

%% End of script

%disp('--> End of script. If you continue, you will lose access to all variables');

%keyboard;