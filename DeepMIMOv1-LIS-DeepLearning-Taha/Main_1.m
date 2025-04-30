clearvars
close all
%clc

base_folder = 'C:/Users/Work/Desktop/deepMIMO/RIS/DeepMIMOv1-LIS-DeepLearning-Taha/';
output_folder = [base_folder, 'Output Matlab/'];
output_folder_py = [base_folder, 'Output_Python/'];

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

network_folder_py = [output_folder_py, 'Neural_Network/'];
figure_folder_py = [output_folder_py, 'Figures/'];
network_folder_out_RateDLpy = [network_folder_py, 'RateDLpy/'];

folders = {DeepMIMO_dataset_folder DL_dataset_folder network_folder network_folder_py figure_folder figure_folder_py network_folder_out_RateDLpy};
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
global load_H_files load_Delta_H_max load_DL_dataset load_Rates training save_mat_files load_mat_py;

load_Delta_H_max = 1; % load output from DeepMIMO_data_generator_2.m
load_H_files     = 1; % load output from DeepMIMO_data_generator_2.m
load_DL_dataset  = 1; % load output from DL_data_generator_3.m
load_Rates       = 1; % load output from DL_training_4.m
training         = 0; % 1 for training the network, 0 from loaoding it from file
save_mat_files   = 0;

load_mat_py      = 3; 
% 3: load py-generated test files
% 2: load py-generated files
% 1: load mat-python-mat files
% 0: load mat-generated files

plot_fig12 = 1;

%plot_fig7 = 0;
plot_figC = 0;

plot_figD = 0;
max_epochs_load = 20;

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
%Training_Size=[10000 30000];

Validation_Size = 6200; % Validation dataset Size
Test_Size = 3100; % Test dataset Size

% fig12
Rate_OPTt=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_mat=zeros(numel(My_ar),numel(Training_Size));  % numel = number of elements

Rate_DLt_py_valOld_20=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_valOld_40=zeros(numel(My_ar),numel(Training_Size));

Rate_DLt_py_val_20=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_val_40=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_val_60=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_val_80=zeros(numel(My_ar),numel(Training_Size));

Rate_DLt_py_test_20=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_test_40=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_test_60=zeros(numel(My_ar),numel(Training_Size));
Rate_DLt_py_test_80=zeros(numel(My_ar),numel(Training_Size));

% figD
MaxR_OPTt = single(zeros(numel(My_ar),numel(Training_Size),Validation_Size));
MaxR_DLt = single(zeros(numel(My_ar),numel(Training_Size),Validation_Size));

MaxR_OPTt_py_test = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_DLt_py_test_20 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_DLt_py_test = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_DLt_py_test_60 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));
MaxR_DLt_py_test_80 = single(zeros(numel(My_ar),numel(Training_Size),Test_Size));

for rr = 1:1:numel(My_ar)

    % Note: The axes of the antennas match the axes of the ray-tracing scenario
    Mx = 1;  % number of LIS reflecting elements across the x axis
    My=My_ar(rr);
    Mz=Mz_ar(rr);

    %% DeepMIMO Dataset Generation

    [Ur_rows_grid]=DeepMIMO_data_generator_2(Mx,My,Mz,D_Lambda,BW,K,K_DL,L,Ut_row,Ut_element,Ur_rows,params);

    %% Deep Learning Dataset Generation = Genie-Aided Reflection Beamforming

    [RandP_all,Validation_Ind]=DL_data_generator_3(Mx,My,Mz,M_bar,D_Lambda,BW,K,K_DL,Pt,Ur_rows,Ut_element,Ur_rows_grid,Validation_Size);

    %Test_Ind = RandP_all2[-Test_Size:] % python
    Test_Ind = RandP_all(end-Test_Size+1:end);
    %Validation_Ind = RandP_all2[-(Validation_Size+Test_Size):-Test_Size] % python
    %Validation_Ind_6200 = RandP_all(end-(Test_Size+Test_Size)+1:end);

    %% DL Beamforming

    for dd=1:1:numel(Training_Size)

        Training_Size_dd = Training_Size(dd);
        end_folder = strcat('_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My), ' ', ''), strrep(num2str(Mz), ' ', ''), '_Mbar', num2str(M_bar));    

        [Rate_OPT,Rate_DL,MaxR_OPT,MaxR_DL]=DL_training_4(Mx,My,Mz,M_bar,Ur_rows,kbeams,Training_Size(dd),RandP_all,Validation_Ind);
        %[Rate_OPT,Rate_DL]=DL_predict_5(Mx,My,Mz,M_bar,Ur_rows,kbeams,Training_Size(dd),RandP_all,Validation_Ind);
        
        if plot_figD == 1
            MaxR_OPTt(rr,dd,:) = MaxR_OPT; % 6200 val
            MaxR_DLt(rr,dd,:) = MaxR_DL; % 6200 val

            % TODO controllare che gli indici di MaxR_DL_py siano nello stesso ordine di quelli di MaxR_DL (matlab).
            end_folder_Training_Size_dd_max_epochs = strcat(end_folder, '_', num2str(Training_Size_dd), '_', num2str(max_epochs_load));
            filename_MaxR_DL_py = strcat(network_folder_out_RateDLpy, 'MaxR_DL_py_test', end_folder_Training_Size_dd_max_epochs, '.mat');
            disp(filename_MaxR_DL_py)
            MaxR_DL_py = h5read(filename_MaxR_DL_py, '/MaxR_DL_py'); % 3100 test
            disp(size(MaxR_DL_py))
            MaxR_DLt_py_test(rr,dd,:) = MaxR_DL_py;
        end

        if plot_fig12 == 1

            Rate_OPTt(rr,dd)=Rate_OPT;
            Rate_DLt_mat(rr,dd)=Rate_DL;

            % Load Rate_DL_py from Python

            plot_types = {'Rate_DL_py_valOld' 'Rate_DL_py_val' 'Rate_DL_py_test'};
            
            for epochs = 20:20:80
            
                for i = 1:length(plot_types)
                    
                    end_folder_Training_Size_dd_max_epochs = strcat(end_folder, '_', num2str(Training_Size_dd), '_', num2str(epochs));
                    filename_Rate_DL_py = strcat(network_folder_out_RateDLpy, plot_types{i}, end_folder_Training_Size_dd_max_epochs, '.mat');
                    
                    try
                        Rate_DL_py = h5read(filename_Rate_DL_py, '/Rate_DL_py');
                    catch exception
                        Rate_DL_py = NaN;
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
                        end

                    end

                end
            end
            
            
            % valOld
            %epochs = 20;
            %end_folder_Training_Size_dd_max_epochs = strcat(end_folder, '_', num2str(Training_Size_dd), '_', num2str(epochs));
            %filename_Rate_DL_py = strcat(network_folder_out_RateDLpy, 'Rate_DL_py_valOld', end_folder_Training_Size_dd_max_epochs, '.mat');
            %Rate_DL_py = h5read(filename_Rate_DL_py, '/Rate_DL_py');
            %Rate_DLt_py_valOld_20(rr,dd)= Rate_DL_py;

            %epochs = 40;
            %end_folder_Training_Size_dd_max_epochs = strcat(end_folder, '_', num2str(Training_Size_dd), '_', num2str(epochs));
            %filename_Rate_DL_py_40 = strcat(network_folder_out_RateDLpy, 'Rate_DL_py_valOld', end_folder_Training_Size_dd_max_epochs, '.mat');
            %%Rate_DL_py = h5read(filename_Rate_DL_py_40, '/Rate_DL_py');
            %Rate_DL_py = 0;
            %Rate_DLt_py_valOld_40(rr,dd)= Rate_DL_py;

            % Val set
            %epochs = 20;
            %end_folder_Training_Size_dd_max_epochs = strcat(end_folder, '_', num2str(Training_Size_dd), '_', num2str(epochs));
            %filename_Rate_DL_py = strcat(network_folder_out_RateDLpy, 'Rate_DL_py_val', end_folder_Training_Size_dd_max_epochs, '.mat');
            %Rate_DL_py = h5read(filename_Rate_DL_py, '/Rate_DL_py');
            %Rate_DLt_py_val_20(rr,dd)= Rate_DL_py;

            % Test set
            %epochs = 20;
            %end_folder_Training_Size_dd_max_epochs = strcat(end_folder, '_', num2str(Training_Size_dd), '_', num2str(epochs));
            %filename_Rate_DL_py = strcat(network_folder_out_RateDLpy, 'Rate_DL_py_test', end_folder_Training_Size_dd_max_epochs, '.mat');
            %Rate_DL_py = h5read(filename_Rate_DL_py, '/Rate_DL_py');
            %Rate_DLt_py_test_20(rr,dd)= Rate_DL_py;

            %epochs = 40;
            %end_folder_Training_Size_dd_max_epochs = strcat(end_folder, '_', num2str(Training_Size_dd), '_', num2str(epochs));
            %filename_Rate_DL_py = strcat(network_folder_out_RateDLpy, 'Rate_DL_py_test', end_folder_Training_Size_dd_max_epochs, '.mat');
            %%Rate_DL_py = h5read(filename_Rate_DL_py, '/Rate_DL_py');
            %Rate_DL_py = 0;
            %Rate_DLt_py_test_40(rr,dd)= Rate_DL_py;
        
            %if My == 64
            %    epochs = 60;
            %    end_folder_Training_Size_dd_max_epochs = strcat(end_folder, '_', num2str(Training_Size_dd), '_', num2str(epochs));
            %    filename_Rate_DL_py_60 = strcat(network_folder_out_RateDLpy, 'Rate_DL_py_test', end_folder_Training_Size_dd_max_epochs, '.mat');
            %    Rate_DL_py = h5read(filename_Rate_DL_py_60, '/Rate_DL_py');
            %    Rate_DLt_py_test_60(rr,dd)= Rate_DL_py;
            %    
            %    epochs = 80;
            %    end_folder_Training_Size_dd_max_epochs = strcat(end_folder, '_', num2str(Training_Size_dd), '_', num2str(epochs));
            %    filename_Rate_DL_py_80 = strcat(network_folder_out_RateDLpy, 'Rate_DL_py_test', end_folder_Training_Size_dd_max_epochs, '.mat');
            %    Rate_DL_py = h5read(filename_Rate_DL_py_80, '/Rate_DL_py');
            %    Rate_DLt_py_test_80(rr,dd)= Rate_DL_py;
            %end
        end
    end

    %keyboard;

end

%% Fig 12
if plot_fig12 == 1
    epochs = 80;
    %Fig12_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,Training_Size,Rate_OPTt,Rate_DLt_mat,Rate_DLt_py_20,Rate_DLt_py_40,Rate_DLt_py_test_20,Rate_DLt_py_test_40,Rate_DLt_py_test_60,Rate_DLt_py_test_80,epochs);
    Fig12_plot(My_ar,Mz_ar,M_bar,Ur_rows,Training_Size,...
                epochs, ...
                Rate_OPTt,Rate_DLt_mat, ...
                Rate_DLt_py_valOld_20,Rate_DLt_py_valOld_40, ...
                Rate_DLt_py_val_20,Rate_DLt_py_test_20, ...
                Rate_DLt_py_val_40,Rate_DLt_py_test_40, ...
                Rate_DLt_py_test_60, ...
                Rate_DLt_py_test_80);

               
end

%% Fig 7 (Luca)

%if plot_fig7 == 1
%    Fig7_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams);
%end

%% Fig C (Luca)

if plot_figC == 1
    % con plot_index=1 si può usare solo plot_test_only=1 (per ora perchè bisogna plottare gli indici il cui rate è superiore alla soglia (non so se è di interesse))
    plot_index     = 0;
    plot_rate      = 1;
    plot_test_only = 1;
    plot_threshold = 1;
    threshold      = 5; % [bps/Hz]
    FigC_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,MaxR_DLt,MaxR_OPTt,Validation_Ind,plot_index,plot_rate,plot_test_only,plot_threshold,threshold);
end

if plot_figD == 1
    plot_mode = 1;
    %FigD_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,MaxR_OPTt,MaxR_DLt,           Validation_Ind,plot_mode)
    %FigD_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,MaxR_OPTt,MaxR_DLt_py_test_20,Test_Ind,      plot_mode)
    plot_mode = 2;
    MaxR_OPTt_py_test = MaxR_OPTt(:,:,end-Test_Size+1:end); % 3100
    FigD_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,MaxR_OPTt_py_test,MaxR_DLt_py_test,Test_Ind,max_epochs_load,plot_mode)
    %keyboard;
    %FigD_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,MaxR_OPTt,MaxR_DLt_py_test_60,Test_Ind,       plot_mode)
    %FigD_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,MaxR_OPTt,MaxR_DLt_py_test_80,Test_Ind,       plot_mode)
end

%% End of script

%disp('--> End of script. If you continue, you will lose access to all variables');

%keyboard;