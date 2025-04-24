function []=FigC_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,MaxR_DLt,MaxR_OPTt,Validation_Ind,plot_index,plot_rate,plot_test_only,plot_threshold,threshold)
%% Description:
%
% This is the function called by the main script for ploting Figure 10 
% in the original article mentioned below.

global load_H_files load_Delta_H_max load_DL_dataset load_Rates training save_mat_files load_mat_py;
global seed DeepMIMO_dataset_folder DL_dataset_folder network_folder network_folder_py figure_folder figure_folder_py;

Training_Size_number=7; % 30000
Training_Size_dd=Training_Size(Training_Size_number);

for rr=1:1:numel(My_ar)

    disp(['---> Plotting FigC with M:', num2str(My_ar(rr)), ', Training_Size:' num2str(Training_Size_dd), ', threshold: ', num2str(threshold), '...']);

    filename_User_Location=strcat(DeepMIMO_dataset_folder, 'User_Location', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');

    %filename_XTrain=strcat(DL_dataset_folder, 'XTrain', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
    %filename_YTrain=strcat(DL_dataset_folder, 'YTrain', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
    filename_XValidation=strcat(DL_dataset_folder, 'XValidation', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
    filename_YValidation=strcat(DL_dataset_folder, 'YValidation', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');

    filename_DL_input_reshaped=strcat(DL_dataset_folder, 'DL_input_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
    filename_DL_output_reshaped=strcat(DL_dataset_folder, 'DL_output_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
    %filename_DL_output_un_reshaped=strcat(DL_dataset_folder, 'DL_output_un_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
    filename_DL_output_un_complete_reshaped=strcat(DL_dataset_folder, 'DL_output_un_complete_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
    filename_trainedNet=strcat(network_folder, 'trainedNet', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
    filename_YPredictedFig7=strcat(network_folder, 'YPredictedFig7', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');

    filename_FigIdx=strcat(figure_folder, 'FigIdx', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.png');
    %filename_FigIdxTh=strcat(figure_folder, 'FigIdxTh', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_th', num2str(threshold), '.png');
    filename_FigIdxTest=strcat(figure_folder, 'FigIdxTest', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.png');
    %filename_FigIdxThTest=strcat(figure_folder, 'FigIdxThTest', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_th', num2str(threshold), '.png');
    
    filename_figRate=strcat(figure_folder, 'FigRate', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.png');
    filename_figRateTh=strcat(figure_folder, 'FigRateTh', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_th', num2str(threshold), '.png');
    filename_figRateTest=strcat(figure_folder, 'FigRateTest', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.png');
    filename_figRateThTest=strcat(figure_folder, 'FigRateThTest', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_th', num2str(threshold), '.png');
    
    filename_figProva=strcat(figure_folder, 'FigProva', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.png');
    
    %%

    x_plot = 1:(Ur_rows(2)-1000);
    y_plot = 1:181;

    if plot_test_only == 1

        disp('plot_test_only == 1')

        if plot_index == 1

            disp('plot_index == 1')

            %load(filename_XTrain);
            %load(filename_YTrain);
            load(filename_XValidation);
            load(filename_YValidation);

            % Recupera gli indici dei codebook
            % Come mai ho usato DL_output_reshaped invece di YValidation per ottenere Indmax_OPT in Fig7?
            % Perchè dovevo plottare tutti gli utenti nella griglia e DL_output_reshaped li contiene tutti,
            % mentre YValidation ne contiene un sottoinsieme.
            [~,Indmax_OPT] = max(YValidation,[],3);
            %disp(['size(Indmax_OPT):', num2str(size(Indmax_OPT))]); % 1, 1, 1, 6200
            Indmax_OPT = squeeze(Indmax_OPT);
            %disp(['size(Indmax_OPT):', num2str(size(Indmax_OPT))]); % 6200, 1
            Indmax_OPT = Indmax_OPT.'; % 1, 6200

            % Carica modello pre-allenato caso completo che copre tutta la grid size
            trainedNet = load(filename_trainedNet).trainedNet;
                
            tic
            YPredictedC = predict(trainedNet,XValidation); % Inferenza sul set di validazione usato come test: errore!
            % Mini-batch size = 128 (default)
            toc
            
            [~,Indmax_DL] = maxk(YPredictedC,kbeams,2);
            %disp(['size(Indmax_DL):', num2str(size(Indmax_DL))]); % 6200, 1

            %keyboard;
        end

        disp('load User_Location')

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
        
        % User_Location è ordinato come DL_input, il quale va in XValidation secondo Validation_Ind:
        %DL_input(:,u+uu-1+((pp-1)*params.num_user)) = reshape([real(H_bar) imag(H_bar)].',[],1); % salva H_bar in
        %User_Location(:,u+uu-1+((pp-1)*params.num_user)) = single(DeepMIMO_dataset{1}.user{u+uu-1}.loc);
        %XValidation = single(DL_input_reshaped(:,1,1,Validation_Ind));
        % Quindi devo ordinare User_Location come XValidation, cioè secondo Validation_Ind
        User_Location_norm_Validation = single(User_Location_norm(:,Validation_Ind));
        %disp(size(User_Location_norm_Validation)) % 3, 6200

        disp('fill sparsity matrix')
        %% Per ogni sample del test set, recupera la sua posizione e il suo rate massimo
        % Se funziona, questo calcolo si può portare dentro a DL_training_4 per semplificare questo script di plot
        values_DL = single(nan(181,200));
        values_OPT = single(nan(181,200));
        for b=1:size(Validation_Ind,1) % 6200
            x = User_Location_norm_Validation(1,b);
            y = User_Location_norm_Validation(2,b);

            if plot_index == 1
                values_OPT(x,y) = Indmax_OPT(b);
                values_DL(x,y) = Indmax_DL(b);
            elseif plot_rate == 1
                % MaxR è ordinato come YValidation_un quindi come DL_output_un, che a sua volta è ordinato secondo Validation_Ind
                %[~,VI_sortind] = sort(Validation_Ind);
                %[~,VI_rev_sortind] = sort(VI_sortind);
                %DL_output_un = DL_output_un(VI_rev_sortind,:);
                %MaxR_DL(b) = squeeze(YValidation_un(1,1,Indmax_DL(b),b));
                % Quindi MaxR è già ordinato secondo Validation_Ind
                values_DL(x,y) = MaxR_DLt(rr,Training_Size_number,b); % equivalent to the previous commented line
            
                %x = User_Location_norm_Validation(1,b);
                %y = User_Location_norm_Validation(2,b);

                values_OPT(x,y) = MaxR_OPTt(rr,Training_Size_number,b);
            end
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

        disp('plot_test_only == 0')

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

        if plot_index == 1 % Come Fig7_plot

            disp('plot_index == 1')

            % Matrice dei valori
            values_OPT_plot = reshape(Indmax_OPT, numel(y_plot), []); % 181x200
            % MATLAB divide il vettore Indmax_OPT in blocchi di lunghezza pari a [] e 
            % li organizza in numel(y_plot) righe.
            % I primi [] elementi di Indmax_OPT diventano la prima riga della matrice.
            % I successivi [] elementi diventano la seconda riga, e così via.
            % Il numero di colonne [] viene calcolato come length(Indmax_OPT) / numel(y_plot).
            values_DL_plot = reshape(Indmax_DL, numel(y_plot), []); % 181x200
    
        elseif plot_rate == 1

            disp('plot_rate == 1')

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

    end




    %% Grafico

    disp('plot fC')

    fC = figure('Name', 'Figure', 'units','pixels', 'Position', [100, 100, 2600, 350]); % typ 800x400

    % Calcolo minimi e massimi colorbar comuni ai due grafici per avere due colorbar uguali
    min_colorbar = min([min(values_OPT_plot(:)), min(values_DL_plot(:))]);
    max_colorbar = max([max(values_OPT_plot(:)), max(values_DL_plot(:))]);

    if plot_rate == 1 && plot_threshold == 1
        % Applica la soglia ai dati
        values_OPT_plot(values_OPT_plot < threshold) = NaN; % Imposta a NaN i valori sotto soglia
        values_DL_plot(values_DL_plot < threshold) = NaN; % Imposta a NaN i valori sotto soglia
        if plot_test_only == 1
            mytitle=['Achievable Rate (Threshold >= ', num2str(threshold), ' bps/Hz), RIS ', num2str(My_ar(rr)), 'x', num2str(My_ar(rr)), ', ', num2str(M_bar), ' active elements (Test Points only)'];
        else
            mytitle=['Achievable Rate (Threshold >= ', num2str(threshold), ' bps/Hz), RIS ', num2str(My_ar(rr)), 'x', num2str(My_ar(rr)), ', ', num2str(M_bar), ' active elements (Training+Test Points)'];
        end
    elseif plot_rate == 1
        if plot_test_only == 1
            mytitle=['Achievable Rate, RIS ', num2str(My_ar(rr)), 'x', num2str(My_ar(rr)), ', ', num2str(M_bar), ' active elements (Test Points only)'];
        else
            mytitle=['Achievable Rate, RIS ', num2str(My_ar(rr)), 'x', num2str(My_ar(rr)), ', ', num2str(M_bar), ' active elements (Training+Test Points)'];
        end
    elseif plot_index == 1
        if plot_test_only == 1
            mytitle=['Codebook index, RIS ', num2str(My_ar(rr)), 'x', num2str(My_ar(rr)), ', ', num2str(M_bar), ' active elements (Test Points only)'];
        else
            mytitle=['Codebook index, RIS ', num2str(My_ar(rr)), 'x', num2str(My_ar(rr)), ', ', num2str(M_bar), ' active elements (Training+Test Points)'];
        end
    end
    sgtitle(mytitle, 'FontWeight', 'bold', 'fontsize', 11);
    disp(mytitle)
    
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
    correct_figC = 1;
    if correct_figC == 0
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

    %% Subplot 1
    subplot(1, 2, 1); % 1 riga, 2 colonne, primo subplot
    
    % Plotta la matrice values mappando ogni elemento a un pixel del grafico.
    %imagesc(values_OPT); 
    imagesc([numel(x_plot), 1], [numel(y_plot), 1], values_OPT_plot); % values(1,1) in basso a destra

    %title('(a) Genie Codebook Beams', 'Units', 'normalized', 'Position', [0.5, -0.2, 0]);
    % 0.5: Centra il titolo orizzontalmente rispetto al subplot.
    % -0.2: Posiziona il titolo sotto il subplot (valore negativo per andare sotto l'asse).
    title('(a) Genie Codebook Beams');

    yticks(y_ticks);
    yticklabels(y_ticks_labels); % The result is {'180', '160', '140', '120', '100', '80', '60', '40', '20', ''}

    % ordinamento corretto assi secondo scenario fig 6
    if correct_figC == 1
        % Questi comandi non vanno bene perchè cambiano anche l'ordine dei dati plottati
        %set(gca, 'XDir', 'reverse'); 
        %set(gca, 'YDir', 'reverse');
        
        xticks(x_ticks_new); % The result is [1, 100, 200]
        xticklabels(x_ticks_labels); % The result is {'1200', '1100', '1000'}

        xlabel('Horizontal user grid direction (y)');
        ylabel('Vertical user grid direction (x)'); 
    else
        xlabel('Horizontal direction (reversed y-axis)');
        ylabel('Vertical direction (reversed x-axis)');
    end

    colormap(parula); % Imposta la colormap arcobaleno
    if plot_test_only == 1 || plot_threshold == 1
        cmap = colormap; % Ottieni la colormap corrente
        colormap([1 1 1; cmap]); % Aggiungi il bianco come primo colore
        set(gca, 'Color', [1 1 1]); % Imposta il colore di sfondo su bianco
    else
        colormap;
    end
    cb = colorbar; % Aggiunge la colorbar e restituisce l'oggetto
    if plot_threshold == 1 || plot_rate == 1
        ylabel(cb, 'Achievable Rate (bps/Hz)','fontsize',11); % Aggiunge una label alla colorbar
    elseif plot_index == 1
        ylabel(cb, 'Codebook index','fontsize',11); % Aggiunge una label alla colorbar
    end
    caxis([min_colorbar, max_colorbar]); % Imposta i limiti della scala dei colori    



    %% Subplot 2
    subplot(1, 2, 2); % 1 riga, 2 colonne, secondo subplot

    imagesc([numel(x_plot), 1], [numel(y_plot), 1], values_DL_plot); % values(1,1) in basso a destra

    title(['(b) DL Codebook Beams (model ', num2str(Training_Size_dd),')']);

    yticks(y_ticks);
    yticklabels(y_ticks_labels); % The result is {'180', '160', '140', '120', '100', '80', '60', '40', '20', ''}

    if correct_figC == 1    
        xticks(x_ticks_new); % The result is [1, 100, 200]
        xticklabels(x_ticks_labels); % The result is {'1200', '1100', '1000'}

        xlabel('Horizontal user grid direction (y)');
        ylabel('Vertical user grid direction (x)'); 
    else
        xlabel('Horizontal direction (reversed y-axis)');
        ylabel('Vertical direction (reversed x-axis)');
    end

    colormap(parula); % Imposta la colormap arcobaleno
    if plot_test_only == 1 || plot_threshold == 1
        cmap = colormap; % Ottieni la colormap corrente
        colormap([1 1 1; cmap]); % Aggiungi il bianco come primo colore
        set(gca, 'Color', [1 1 1]); % Imposta il colore di sfondo su bianco
    else
        colormap;
    end
    cb = colorbar; % Aggiunge la colorbar e restituisce l'oggetto
    if plot_threshold == 1 || plot_rate == 1
        ylabel(cb, 'Achievable Rate (bps/Hz)','fontsize',11); % Aggiunge una label alla colorbar
    elseif plot_index == 1
        ylabel(cb, 'Codebook index','fontsize',11); % Aggiunge una label alla colorbar
    end
    caxis([min_colorbar, max_colorbar]); % Imposta i limiti della scala dei colori  

    %% Save fig
    if plot_rate == 1
        if plot_threshold == 1
            if plot_test_only == 1
                saveas(fC, filename_figRateThTest);
            else
                saveas(fC, filename_figRateTh);
            end
        else 
            if plot_test_only == 1
                saveas(fC, filename_figRateTest);
            else
                saveas(fC, filename_figRate);
            end
        end
    elseif plot_index == 1
        if plot_test_only == 1
            saveas(fC, filename_FigIdxTest);
        else
            saveas(fC, filename_FigIdx);
        end
    end
    close(fC);

    disp('Done');


    %fC = figure('Name', 'Figure', 'units','pixels', 'Position', [100, 100, 2600, 350]); % typ 800x400
    %imagesc(values_DL_plot);
    %colorbar;
    %saveas(fC, filename_figProva);

    %keyboard;

end