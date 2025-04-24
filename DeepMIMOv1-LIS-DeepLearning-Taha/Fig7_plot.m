function []=Fig7_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams)
%% Description:
%
% This is the function called by the main script for ploting Figure 10 
% in the original article mentioned below.

global load_H_files load_Delta_H_max load_DL_dataset load_Rates training save_mat_files load_mat_py;
global seed DeepMIMO_dataset_folder DL_dataset_folder network_folder network_folder_py figure_folder figure_folder_py;

Training_Size=30000;

for i=1:1:2
    
    correct_fig7 = i-1;

    for rr=1:1:numel(My_ar)

        disp(['---> Plotting Fig7 with ', num2str(My_ar(rr)), ' and ' num2str(correct_fig7), '...']);

        filename_DL_input_reshaped=strcat(DL_dataset_folder, 'DL_input_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
        filename_DL_output_reshaped=strcat(DL_dataset_folder, 'DL_output_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
        filename_trainedNet=strcat(network_folder, 'trainedNet', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size), '.mat');        
        filename_YPredictedFig7=strcat(network_folder, 'YPredictedFig7', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
        filename_YPredictedFig7_mat=strcat(network_folder_py, 'YPredictedFig7_mat', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');

        filename_fig7=strcat(figure_folder, 'Fig7', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_correct', num2str(correct_fig7), '.png');
        filename_fig7_mat=strcat(figure_folder_py, 'Fig7_mat', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_correct', num2str(correct_fig7), '.png');        

        % Concatena YTrain + YValidation, cioè utilizza DL_output_reshaped
        load(filename_DL_output_reshaped);

        %%%%%% TEMP
        % Concatena XTrain + XValidation, cioè utilizza DL_input_reshaped
        load(filename_DL_input_reshaped);
        
        % Carica modello pre-allenato caso completo che copre tutta la grid size
        trainedNet = load(filename_trainedNet).trainedNet;
        
        % Esegui predizione con DL_input_reshaped
        tic
        disp('Start DL prediction for Figure 7...')
        YPredictedFig7_orig = predict(trainedNet,DL_input_reshaped); % Inferenza sul set di validazione usato come test: errore!
        disp('Done')
        toc
        
        [~,Indmax_DL_orig] = maxk(YPredictedFig7_orig,kbeams,2);
        %%%%%% TEMP

        if save_mat_files == 1

            % Concatena XTrain + XValidation, cioè utilizza DL_input_reshaped
            load(filename_DL_input_reshaped);

            % Carica modello pre-allenato caso completo che copre tutta la grid size
            trainedNet = load(filename_trainedNet).trainedNet;
            
            % Esegui predizione con DL_input_reshaped
            tic
            disp('Start DL prediction for Figure 7...')
            YPredictedFig7 = predict(trainedNet,DL_input_reshaped); % Inferenza sul set di validazione usato come test: errore!
            % Mini-batch size = 128 (default)
            disp('Done')
            toc
            save(filename_YPredictedFig7,'YPredictedFig7','-v7.3'); 
        else
            if load_mat_py == 1
                disp('Import YPredictedFig7 from Python')
                YPredictedFig7 = h5read(filename_YPredictedFig7_mat, '/YPredictedFig7_mat');
                YPredictedFig7 = YPredictedFig7'; % Transpose perchè per la disposizione dei dati in memoria:
                % HDF5 (e quindi h5py) usa la convenzione row-major (C-style)
                % mentre MATLAB usa column-major (Fortran-style)
                disp('Done')
            elseif load_mat_py == 0
                load(filename_YPredictedFig7);
            end
        end

        % Recupera gli indici dei codebook
        % Come mai ho usato DL_output_reshaped invece di YValidation per ottenere Indmax_OPT in Fig7?
        % Perchè dovevo plottare tutti gli utenti nella griglia e DL_output_reshaped li contiene tutti,
        % mentre YValidation ne contiene un sottoinsieme.
        [~,Indmax_OPT] = max(DL_output_reshaped,[],3);
        %disp(['size(Indmax_OPT):', num2str(size(Indmax_OPT))]); % 1, 1, 1, 36200
        Indmax_OPT = squeeze(Indmax_OPT);
        %disp(['size(Indmax_OPT):', num2str(size(Indmax_OPT))]); % 36200, 1
        Indmax_OPT = Indmax_OPT.'; % 1, 36200

        [~,Indmax_DL] = maxk(YPredictedFig7,kbeams,2);
        %disp(['size(Indmax_DL):', num2str(size(Indmax_DL))]); % 36200, 1

        %%%%%% TEMP
        tolerance = 1;
        diff = abs(Indmax_DL(:) - Indmax_DL_orig(:));
        differenceMask = diff >= tolerance;
        %areEqual = all(diff <= tolerance);
        
        Indmax_DL_diff = Indmax_DL(differenceMask);
        Indmax_DL_orig_diff = Indmax_DL_orig(differenceMask);
        
        for i = 1:length(Indmax_DL_diff)
            fprintf('Indmax_DL_diff = %d, Indmax_DL_orig_diff = %d\n', Indmax_DL_diff(i), Indmax_DL_orig_diff(i));
        end
        %%%%%% TEMP

        %keyboard;

        % Grafico
        x = 1:(Ur_rows(2)-1000);
        y = 1:181;

        % Matrice dei valori
        values_OPT = reshape(Indmax_OPT, numel(y), []); % 181x200
        % MATLAB divide il vettore Indmax_OPT in blocchi di lunghezza pari a [] e 
        % li organizza in numel(y) righe.
        % I primi [] elementi di Indmax_OPT diventano la prima riga della matrice.
        % I successivi [] elementi diventano la seconda riga, e così via.
        % Il numero di colonne [] viene calcolato come length(Indmax_OPT) / numel(y).
        values_DL = reshape(Indmax_DL, numel(y), []); % 181x200

        f7 = figure('Name', 'Figure7', 'units','pixels', 'Position', [100, 100, 2600, 300]); % typ 800x400

        % Calcolo minimi e massimi colorbar comuni ai due grafici per avere due colorbar uguali
        min_colorbar = min([min(values_OPT(:)), min(values_DL(:))]);
        max_colorbar = max([max(values_OPT(:)), max(values_DL(:))]);

        % Test imagesc
        % Di default imagesc plotta nell'angolo in alto a sinistra il primo elemento dell'immagine di ingresso c(1,1)
        %c = reshape(1:1:100, 10, 10).';
        %imagesc(c); 
        % Per spostare il punto in cui vogliamo spostare il valore c(1,1), bisogna indicare le coordinate di dove vogliamo
        % il primo punto e l'ultimo punto della matrice c, nell'esempio (10,10) e (1,1), rispettivamente.
        % Per trovare quali numeri mettere, bisogna prima plottare la matrice senza range, e poi inserirli successivamente
        % utilizzando i valori che sono riportati sugli assi della prima figura come sistema di riferimento.
        %c = reshape(1:1:100, 10, 10).';
        %imagesc([10, 1], [10, 1], c); 

        if correct_fig7 == 0
            offset=-2; % Per avere un tick in alto per far vedere il 181 utente
            y_ticks = (1+offset):20:(181+offset);
            y_ticks_notflipped = y_ticks-offset-1; % Ordina l'array al contrario
            y_ticks_string = string(y_ticks_notflipped); % Converte i valori in un array di stringhe
            y_ticks_labels = cellstr(y_ticks_string); % Converte in un array di celle
            %y_ticks_labels(end) = '';
        else
            x_ticks = Ur_rows(1):100:(Ur_rows(2));
            x_ticks_new = x_ticks - 1000;
            x_ticks_new(1) = 1;

            x_ticks_flipped = fliplr(x_ticks); % Ordina l'array al contrario
            x_ticks_string = string(x_ticks_flipped); % Converte i valori in un array di stringhe
            x_ticks_labels = cellstr(x_ticks_string); % Converte in un array di celle

            offset=3; % Per avere un tick in alto per far vedere il 181 utente
            y_ticks = (1+offset):20:(181+offset);
            y_ticks_flipped = fliplr(y_ticks-offset-1); % Ordina l'array al contrario
            y_ticks_string = string(y_ticks_flipped); % Converte i valori in un array di stringhe
            y_ticks_labels = cellstr(y_ticks_string); % Converte in un array di celle
            y_ticks_labels(end) = '';
        end

        % Subplot 1
        subplot(1, 2, 1); % 1 riga, 2 colonne, primo subplot
        title('(a) Original Codebook Beams');

        % Plotta la matrice values mappando ogni elemento a un pixel del grafico.
        %imagesc(values_OPT); 
        imagesc([numel(x), 1], [numel(y), 1], values_OPT); % values(1,1) in basso a destra

        yticks(y_ticks);
        yticklabels(y_ticks_labels); % The result is {'180', '160', '140', '120', '100', '80', '60', '40', '20', ''}

        xlabel('Horizontal direction (reversed y-axis)');
        ylabel('Vertical direction (reversed x-axis)');


        % ordinamento corretto assi secondo scenario fig 6
        if correct_fig7 == 1
            % Questi comandi non vanno bene perchè cambiano anche l'ordine dei dati plottati
            %set(gca, 'XDir', 'reverse'); 
            %set(gca, 'YDir', 'reverse');
            
            xticks(x_ticks_new); % The result is [1, 100, 200]
            xticklabels(x_ticks_labels); % The result is {'1200', '1100', '1000'}

            xlabel('Horizontal direction');
            ylabel('Vertical direction'); 
        end

        colormap(parula); % Imposta la colormap arcobaleno
        colorbar; % Aggiunge una barra dei colori
        caxis([min_colorbar, max_colorbar]); % Imposta i limiti della scala dei colori



        % Subplot 2
        subplot(1, 2, 2); % 1 riga, 2 colonne, secondo subplot
        title('(b) Predicted Codebook Beams');

        imagesc([numel(x), 1], [numel(y), 1], values_DL); % values(1,1) in basso a destra

        yticks(y_ticks);
        yticklabels(y_ticks_labels); % The result is {'180', '160', '140', '120', '100', '80', '60', '40', '20', ''}

        xlabel('Horizontal direction (reversed y-axis)');
        ylabel('Vertical direction (reversed x-axis)');

        if correct_fig7 == 1    
            xticks(x_ticks_new); % The result is [1, 100, 200]
            xticklabels(x_ticks_labels); % The result is {'1200', '1100', '1000'}

            xlabel('Horizontal direction');
            ylabel('Vertical direction');   
        end

        colormap(parula); % Imposta la colormap arcobaleno
        colorbar; % Aggiunge una barra dei colori
        caxis([min_colorbar, max_colorbar]); % Imposta i limiti della scala dei colori

        if load_mat_py == 1
            saveas(f7, filename_fig7_mat);
        elseif load_mat_py == 0
            saveas(f7, filename_fig7);
        end
        
        close(f7);

        disp('Done');

        %keyboard;

    end
end





%% OLD

% Matrice dei valori

% Recupera locations loc
%load(filename_User_Location);
%
%User_Location_norm = single(zeros(3,No_user_pairs));
%space_between_users = 0.2; % meters
%for i=1:1:size(User_Location_norm,1)-1 % Lavora solo su x e y perchè z è costante
%    User_Location_norm(i, :) = round(( User_Location(i,:) - min(User_Location(i,:)) ) / space_between_users + 1);
%    %disp(min(User_Location(i,:)));
%    %disp(min(User_Location_norm(i,:)));
%    %disp(max(User_Location(i,:)));
%    %disp(max(User_Location_norm(i,:)));
%end
%keyboard;
    
% Grafico
% Il sistema di riferimento x,y è quello del grafico, NON quello dello scenario.
%reversed_y = 1:200; %User_Location_norm(2, 1:181:end); % prendere un valore ogni 181, risultando in un vettore da 1 a 200
%reversed_x = 1:181; %User_Location_norm(1, 1:max(User_Location_norm(1,:))); % prendere i primi 181 valori, risultando in un vettore da 1 a 181

%{
values = single(zeros(numel(reversed_x), numel(reversed_y))); % 181x200
% Deve essere una matrice x x y, cioè 181x200
% Devo cambiare la shape di questa matrice in modo che in una riga ci siano gli indici da 1 a 200.
% Adesso invece in una riga ci sono prima gli indici da 1 a 100 del primo rettangolo di utenti,
% poi ci sono gli indici da 18101 a 18200 del secondo rettangolo di utenti (vedi foglio).

try
    load(filename_params);
    disp('Load params.num_user')
catch ME
    disp(params.num_user);
    params.num_user = u_step*181;
    disp(params.num_user);
    disp('Fix params.num_user');
end

i=0;
for pp = 1:1:numel(Ur_rows_grid)-1 % Per ogni regione verticale
    for u=1:u_step:params.num_user % Per ogni utente dentro una regione verticale a step di 100 (u_step)

        % DL_output(u+((pp-1)*params.num_user):u+((pp-1)*params.num_user)+u_step-1,:) = Rn;

        r = int32( (i+1)/2 ); % ogni due for loop interni, incrementa di 1
        m = mod(i,2); % alterna i valori 0 e 1
        c = u_step*m+1:u_step*m+u_step; % 1:100 when m=0, 101:200 when m=1
        one_hundred_users = ( Indmax_OPT(u+((pp-1)*params.num_user):u+((pp-1)*params.num_user)+u_step-1) ).';
        %one_hundred_users = Indmax_OPT;
        % .' is for chaning the shape from column to row vector
        size(one_hundred_users);

        values(r, c) = one_hundred_users;
        % Ad ogni riga leggo 100 valori
        
        i=i+1;

        %disp(['r=', num2str(r)]);
        %disp(['c=', num2str(c)]);
        %disp(['i=', num2str(i)]);
        %disp('---')

        %keyboard;
    end
end
%}

%{
values2 = single(zeros(numel(reversed_x), numel(reversed_y))); % 181x200

try
    load(filename_params);
    disp('Load params.num_user')
catch ME
    %disp(params.num_user);
    params.num_user = u_step*181;
    %disp(params.num_user);
    disp('Fix params.num_user');
end

for pp = 1:1:numel(Ur_rows_grid)-1 % Per ogni regione verticale
    r=1;
    m = mod(pp,1); % if pp==1, m=0; if pp==2, m=1
    c = u_step*m+1:u_step*m+u_step; % 1:100 when m=0, 101:200 when m=1
    for u=1:u_step:params.num_user % Per ogni utente dentro una regione verticale a step di 100 (u_step)

        % DL_output(u+((pp-1)*params.num_user):u+((pp-1)*params.num_user)+u_step-1,:) = Rn;

        one_hundred_users = Indmax_OPT(1, u+((pp-1)*params.num_user):u+((pp-1)*params.num_user)+u_step-1);

        values2(r, c) = one_hundred_users;
        % Ad ogni riga leggo 100 valori

        %disp(u+((pp-1)*params.num_user));
        %disp(['r=', num2str(r)]);
        %disp(['c=', num2str(c)]);
        %disp('---')

        r=r+1;

        %keyboard;
    end
end

values3 = reshape(Indmax_OPT, u_step, []).'; % 362, 100
values3 = [values3(1:181, :), values3(182:end, :)]; %181, 200
% Matrice 181x200 spostando le righe dalla 182 alla fine di values2 accanto alle prime 181 righe.

disp(isequal(values1, values2));
disp(isequal(values2, values3));
%}