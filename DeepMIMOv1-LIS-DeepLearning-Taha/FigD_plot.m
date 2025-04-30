function []=FigD_plot(My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,Validation_Ind,Test_Ind,epochs,plot_mode, ...
                        MaxR_OPTt,MaxR_DLt_mat, ...
                        MaxR_OPTt_py_test_20,MaxR_DLt_py_val_20,MaxR_DLt_py_test_20, ...
                        MaxR_OPTt_py_test_40,MaxR_DLt_py_val_40,MaxR_DLt_py_test_40, ...
                        MaxR_OPTt_py_test_60,MaxR_DLt_py_val_60,MaxR_DLt_py_test_60, ...
                        MaxR_OPTt_py_test_80,MaxR_DLt_py_val_80,MaxR_DLt_py_test_80)
%% Description:
%
% This is the function called by the main script for ploting Figure 10 
% in the original article mentioned below.

global load_H_files load_Delta_H_max load_DL_dataset load_Rates training save_mat_files load_mat_py;
global seed DeepMIMO_dataset_folder DL_dataset_folder network_folder network_folder_py figure_folder figure_folder_py;

% TODO BETTER with HOLD ON like in Fig12

if plot_mode == 2
    MaxR_OPTt = MaxR_OPTt_py_test_20;
    MaxR_DLt = MaxR_DLt_py_test_20;
elseif plot_mode == 1
    MaxR_OPTt = MaxR_OPTt;
    MaxR_DLt = MaxR_DLt_mat;
end

if plot_mode == 1 && epochs ~= 20
    disp('ERROR: for plot_mode == 1 epochs can only be 20')
    exit;
end
    
filename_figD=strcat(figure_folder, 'FigD', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '.png');
filename_figDmatVal=strcat(figure_folder_py, 'FigDmatVal', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '_', num2str(epochs), '.png');
filename_figDpyTest=strcat(figure_folder_py, 'FigDpyTest', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '_', num2str(epochs), '.png');
    
Training_Size_number=7; % 30000
Training_Size_dd=Training_Size(Training_Size_number);

th_step=0.5;

if epochs < 60
    Colour = 'brgmcky';
else
    Colour = 'rbgmcky';
end
Marker = {'--o', '-o'; '--square', '-square'};

fD = figure('Name', 'Figure', 'units','pixels', 'Position', [100, 100, 1200, 400]); % typ 800x400
hold on; grid on; box on;
xlabel('Rate Threshold (bps/Hz)','fontsize',10)
ylabel('Coverage (%)','fontsize',10)
set(gca,'FontSize',11)

for rr=1:1:numel(My_ar)

    % TODO new title
    if plot_mode == 2
        title(['Coverage vs Rate Threshold, RIS ', num2str(M_bar), ' active elements, pyTest model trained with ', num2str(Training_Size_dd), ' samples and ', num2str(epochs), ' epochs'],'fontsize',11)
    elseif plot_mode == 1
        title(['Coverage vs Rate Threshold, RIS ', num2str(M_bar), ' active elements, matVal model trained with ', num2str(Training_Size_dd), ' samples and ', num2str(epochs), ' epochs'],'fontsize',11)
    elseif plot_mode == 0
        title(['Coverage vs Rate Threshold, RIS ', num2str(M_bar), ' active elements, model ', num2str(Training_Size_dd), ' epochs ', num2str(epochs), ' (Training+Old Val points)'],'fontsize',11)
    end

    disp(['---> Plotting FigD with M:', num2str(My_ar(rr)), ', Training_Size:' num2str(Training_Size_dd), ', epochs:' num2str(epochs), '...']);

    %%

    x_plot = 1:(Ur_rows(2)-1000);
    y_plot = 1:181;

    if plot_mode == 2 || plot_mode == 1

        disp(['plot_mode == ', num2str(plot_mode)])

        disp('load User_Location')
        filename_User_Location=strcat(DeepMIMO_dataset_folder, 'User_Location', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
        load(filename_User_Location);
        %disp(size(User_Location)) % 3, 36200

        User_Location_norm = single(zeros(3,size(User_Location,2)));
        space_between_users = 0.2; % meters
        for i=1:1:2 % Lavora solo su x e y perchè z è costante
            User_Location_norm(i, :) = round(( User_Location(i,:) - min(User_Location(i,:)) ) / space_between_users + 1);
        end
        %disp(min(User_Location(1,:)));
        %disp(min(User_Location_norm(1,:)));
        %disp(max(User_Location(1,:)));
        %disp(max(User_Location_norm(1,:)));
        %disp(min(User_Location(2,:)));
        %disp(min(User_Location_norm(2,:)));
        %disp(max(User_Location(2,:)));
        %disp(max(User_Location_norm(2,:)));
        
        % User_Location è ordinato come DL_input, il quale va in XValidation secondo ValTest_Ind:
        %DL_input(:,u+uu-1+((pp-1)*params.num_user)) = reshape([real(H_bar) imag(H_bar)].',[],1); % salva H_bar in
        %User_Location(:,u+uu-1+((pp-1)*params.num_user)) = single(DeepMIMO_dataset{1}.user{u+uu-1}.loc);
        %XValidation = single(DL_input_reshaped(:,1,1,ValTest_Ind));
        % Quindi devo ordinare User_Location come XValidation, cioè secondo ValTest_Ind
        if plot_mode == 2
            User_Location_norm_ValTest = single(User_Location_norm(:,Test_Ind));
            ValTest_Ind = Test_Ind;
        elseif plot_mode == 1
            User_Location_norm_ValTest = single(User_Location_norm(:,Validation_Ind));
            ValTest_Ind = Validation_Ind;
        end
        %disp(size(User_Location_norm_ValTest)) % 3, 6200

        disp('fill sparsity matrix')
        %% Per ogni sample del test set, recupera la sua posizione e il suo rate massimo
        % Se funziona, questo calcolo si può portare dentro a DL_training_4 per semplificare questo script di plot
        values_DL = single(nan(181,200));
        values_OPT = single(nan(181,200));
        for b=1:size(ValTest_Ind,1) % 6200 val o 3100 test
            x = User_Location_norm_ValTest(1,b);
            y = User_Location_norm_ValTest(2,b);

            % MaxR è ordinato come YValidation_un quindi come DL_output_un, che a sua volta è ordinato secondo ValTest_Ind
            %[~,VI_sortind] = sort(ValTest_Ind);
            %[~,VI_rev_sortind] = sort(VI_sortind);
            %DL_output_un = DL_output_un(VI_rev_sortind,:);
            %MaxR_DL(b) = squeeze(YValidation_un(1,1,Indmax_DL(b),b));
            % Quindi MaxR è già ordinato secondo ValTest_Ind
            values_DL(x,y) = MaxR_DLt(rr,Training_Size_number,b); % equivalent to the previous commented line
        
            %x = User_Location_norm_ValTest(1,b);
            %y = User_Location_norm_ValTest(2,b);

            values_OPT(x,y) = MaxR_OPTt(rr,Training_Size_number,b);
        end

        % Matrice dei valori
        values_OPT_plot = reshape(values_OPT, numel(y_plot), []); % 181x200
        values_DL_plot = reshape(values_DL, numel(y_plot), []); % 181x200
        % MATLAB divide il vettore Indmax_OPT in blocchi di lunghezza pari a [] e 
        % li organizza in numel(y_plot) righe.
        % I primi [] elementi di Indmax_OPT diventano la prima riga della matrice.
        % I successivi [] elementi diventano la seconda riga, e così via.
        % Il numero di colonne [] viene calcolato come length(Indmax_OPT) / numel(y_plot).

        %if numel(Indmax_DL) == numel(unique(Indmax_DL))
        %    disp('L\'array contiene solo numeri univoci.');
        %else
        %    disp('L\'array contiene numeri duplicati.');
        %end

        %keyboard;
    else

        disp('plot_mode == 0')
        disp("ATTENZIONE!!! QUESTO E' CODICE VECCHIO: TODO")

        %filename_XTrain=strcat(DL_dataset_folder, 'XTrain', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
        %filename_YTrain=strcat(DL_dataset_folder, 'YTrain', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
        %filename_XValidation=strcat(DL_dataset_folder, 'XValidation', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
        %filename_YValidation=strcat(DL_dataset_folder, 'YValidation', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');

        filename_DL_input_reshaped=strcat(DL_dataset_folder, 'DL_input_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
        filename_DL_output_reshaped=strcat(DL_dataset_folder, 'DL_output_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
        %filename_DL_output_un_reshaped=strcat(DL_dataset_folder, 'DL_output_un_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
        filename_DL_output_un_complete_reshaped=strcat(DL_dataset_folder, 'DL_output_un_complete_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
        filename_trainedNet=strcat(network_folder, 'trainedNet', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
        filename_YPredictedFig7=strcat(network_folder, 'YPredictedFig7', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');

        % Concatena YTrain + YValidation, cioè utilizza DL_output_reshaped
        load(filename_DL_output_reshaped);

        %%%%%% TEMP
        % Concatena XTrain + XValidation, cioè utilizza DL_input_reshaped
        load(filename_DL_input_reshaped);
        
        % Carica modello pre-allenato caso completo che copre tutta la grid size
        trainedNet = load(filename_trainedNet).trainedNet;
        
        % Esegui predizione con DL_input_reshaped
        %tic
        %disp('Start DL prediction for Figure 7...')
        %YPredictedFig7 = predict(trainedNet,DL_input_reshaped); % Inferenza sul set di validazione usato come test: errore!
        %disp('Done')
        %toc
        load(filename_YPredictedFig7);
        
        [~,Indmax_DL] = maxk(YPredictedFig7,kbeams,2); % 36200, 1

        % Recupera gli indici dei codebook
        % Come mai ho usato DL_output_reshaped invece di YValidation per ottenere Indmax_OPT in Fig7?
        % Perchè dovevo plottare tutti gli utenti nella griglia e DL_output_reshaped li contiene tutti,
        % mentre YValidation ne contiene un sottoinsieme.
        [~,Indmax_OPT] = max(DL_output_reshaped,[],3);
        %disp(['size(Indmax_OPT):', num2str(size(Indmax_OPT))]); % 1, 1, 1, 36200
        Indmax_OPT = squeeze(Indmax_OPT);
        %disp(['size(Indmax_OPT):', num2str(size(Indmax_OPT))]); % 36200, 1
        Indmax_OPT = Indmax_OPT.'; % 1, 36200

        %MaxR_OPTt(rr,Training_Size_number,b);
        % TODO
        load(filename_DL_output_un_complete_reshaped);
        YValidation_un_complete = single(DL_output_un_complete_reshaped);

        %disp(size(YValidation_un_complete));

        MaxR_DL = single(zeros(size(Indmax_DL,1),1));
        MaxR_OPT = single(zeros(numel(Indmax_OPT),1));

        %keyboard;

        for b=1:size(Indmax_DL,1) % 36200
            % YValidation_un = DL_output_un_reshaped = DL_output_un
            MaxR_DL(b) = squeeze(YValidation_un_complete(1,1,Indmax_DL(b),b));
            MaxR_OPT(b) = squeeze(YValidation_un_complete(1,1,Indmax_OPT(b),b));
        end

        values_OPT_plot = reshape(MaxR_OPT, numel(y_plot), []); % 181x200
        values_DL_plot = reshape(MaxR_DL, numel(y_plot), []); % 181x200

        %disp(size(values_OPT_plot))

        %keyboard;

    end

    %% Grafico

    disp('plot fD ...')

    % Applica la soglia ai dati
    disp(max(values_OPT_plot(:)))
    disp(max(values_DL_plot(:)))
    max_colorbar = max([max(values_OPT_plot(:)), max(values_DL_plot(:))]);
    %rounded_max = ceil(max_colorbar * 10) / 10;
    rounded_max = floor(max_colorbar+1);
    disp(max_colorbar)
    disp(rounded_max)

    punti_coperti_sul_totale_OPT = single(zeros(1,rounded_max/th_step+1));
    punti_coperti_sul_totale_DL = single(zeros(1,rounded_max/th_step+1));
    thresholds = single(0:th_step:rounded_max);

    %disp(numel(punti_coperti_sul_totale_OPT))
    %disp(numel(thresholds))

    if plot_mode == 2 || plot_mode == 1
        numero_punti = size(ValTest_Ind,1); % 6200 o 3100
    elseif plot_mode == 0
        numero_punti = numel(values_OPT_plot); % 36200
    end

    for i=1:numel(thresholds)
        % OPT
        punti_coperti = sum(values_OPT_plot(:) >= thresholds(i));
        punti_coperti_sul_totale_OPT(i) = punti_coperti/numero_punti*100;
        disp([num2str(thresholds(i)), ' ', num2str(punti_coperti), ' ', num2str(punti_coperti_sul_totale_OPT(i))])
        % DL
        punti_coperti = sum(values_DL_plot(:) >= thresholds(i));
        punti_coperti_sul_totale_DL(i) = punti_coperti/numero_punti*100;
        disp([num2str(thresholds(i)), ' ', num2str(punti_coperti), ' ', num2str(punti_coperti_sul_totale_DL(i))])
    end

    plot(thresholds,punti_coperti_sul_totale_OPT,[Colour(rr) Marker{rr, 1}],'markersize',7,'linewidth',1., 'DisplayName',['Genie,        M = ' num2str(My_ar(rr)) 'x' num2str(Mz_ar(rr))])
    
    if plot_mode == 2
        plot(thresholds,punti_coperti_sul_totale_DL,[Colour(rr) Marker{rr, 2}],'markersize',7,'linewidth',1., 'DisplayName', ['DL_{pyTest' num2str(epochs) '}, M = ' num2str(My_ar(rr)) 'x' num2str(Mz_ar(rr))])
    elseif plot_mode == 1
        plot(thresholds,punti_coperti_sul_totale_DL,[Colour(rr) Marker{rr, 2}],'markersize',7,'linewidth',1., 'DisplayName', ['DL_{matVal20}, M = ' num2str(My_ar(rr)) 'x' num2str(Mz_ar(rr))])
    end

    %fD = figure('Name', 'Figure', 'units','pixels', 'Position', [100, 100, 2600, 350]); % typ 800x400
    %imagesc(values_DL_plot);
    %colorbar;
    %saveas(fD, filename_figProva);

    %keyboard;

end

%legend('Location','SouthEast')
lgd = legend('Location','northeast');
fontsize(lgd,9,'points')
legend show
ylim([0 102]);

%% Save fig
if plot_mode == 2
    saveas(fD, filename_figDpyTest);
elseif plot_mode == 1
    saveas(fD, filename_figDmatVal);
elseif plot_mode == 0
    saveas(fD, filename_figDTest);
end
close(fD);

disp('Done');