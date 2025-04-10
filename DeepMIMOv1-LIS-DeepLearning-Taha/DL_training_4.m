function [Rate_OPT,Rate_DL]=DL_training_4(Mx,My,Mz,M_bar,Ur_rows,kbeams,Training_Size_dd,RandP_all,Validation_Ind)

%% DL Beamforming

disp(['---> DL Beamforming for Training_Size ', num2str(Training_Size_dd)]);

global load_H_files load_Delta_H_max load_DL_dataset load_Rates training save_mat_files load_mat;
global seed DeepMIMO_dataset_folder DL_dataset_folder network_folder network_folder_py figure_folder figure_folder_py;

filename_DL_input_reshaped=strcat(DL_dataset_folder, 'DL_input_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '.mat');
filename_DL_output_reshaped=strcat(DL_dataset_folder, 'DL_output_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '.mat');
filename_DL_output_un_reshaped=strcat(DL_dataset_folder, 'DL_output_un_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '.mat');

filename_XTrain=strcat(DL_dataset_folder, 'XTrain', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
filename_YTrain=strcat(DL_dataset_folder, 'YTrain', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
filename_XValidation=strcat(DL_dataset_folder, 'XValidation', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
filename_YValidation=strcat(DL_dataset_folder, 'YValidation', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');

filename_trainedNet=strcat(network_folder, 'trainedNet', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
filename_trainedNet_tf=strcat(network_folder, 'trainedNet', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd));
filename_trainedNet_scaler=strcat(network_folder, 'trainedNet_scaler', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
%filename_YPredicted=strcat(network_folder, 'YPredicted', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '.mat');
filename_YPredicted_mat=strcat(network_folder_py, 'YPredicted_mat', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
filename_Rate_DL=strcat(network_folder, 'Rate_DL', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
filename_Rate_OPT=strcat(network_folder, 'Rate_OPT', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
filename_Rate_DL_mat=strcat(network_folder_py, 'Rate_DL_mat', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
filename_Rate_OPT_mat=strcat(network_folder_py, 'Rate_OPT_mat', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');

if load_Rates == 1
    if load_mat == 1
        disp('Loading Rate_DL_mat, Rate_OPT_mat...');
        load(filename_Rate_DL_mat);
        load(filename_Rate_OPT_mat);
        disp('Done');
    else
        disp('Loading Rate_DL, Rate_OPT...');
        load(filename_Rate_DL);
        load(filename_Rate_OPT);
        disp('Done');
    end

    %keyboard;
else

    disp('Loading DL_input and DL_output...');
    load(filename_DL_input_reshaped);
    load(filename_DL_output_reshaped);
    load(filename_DL_output_un_reshaped);
    disp('Done');

    miniBatchSize  = 500; % Size of the minibatch for the Deep Learning

    % Preallocation of output variables
    %Rate_DL = zeros(1,length(Training_Size)); 
    Rate_DL = 0;
    Rate_OPT = Rate_DL;
    LastValidationRMSE = Rate_DL;
    %Rate_DL_fake = Rate_DL; % Luca
    validation_accuracy = Rate_DL; % Luca

    % ------------------ Training and Testing Datasets -----------------%
    % Per ogni punto del grafico, viene allenato un modello
    %for dd=1:1:numel(Training_Size)
    %disp([' Calculating for Dataset Size = ' num2str(Training_Size(dd))]);
    disp([' Calculating for Dataset Size = ' num2str(Training_Size_dd)]);
    % Get a random number of indeces equal to the content of one element of the Training_Size array
    %Training_Ind   = RandP_all(1:Training_Size(dd));
    Training_Ind   = RandP_all(1:Training_Size_dd);

    % Use the indexes to extract the actual sampled used for training from DL_input.
    % This is why DL_input is designed to be No_user_pairs log while in reality it is shorter and equal to Training_Ind = Training_Size(dd).
    XTrain = single(DL_input_reshaped(:,1,1,Training_Ind));
    YTrain = single(DL_output_reshaped(1,1,:,Training_Ind));
    XValidation = single(DL_input_reshaped(:,1,1,Validation_Ind));
    YValidation = single(DL_output_reshaped(1,1,:,Validation_Ind));
    YValidation_un = single(DL_output_un_reshaped);

    %disp([' size(XTrain) = ' num2str(size(XTrain))]); % 1024, 1, 1, Training_Ind(dd)
    %disp([' size(YTrain) = ' num2str(size(YTrain))]); % 1, 1, 1024, Training_Ind(dd)
    %disp([' size(XValidation) = ' num2str(size(XValidation))]); % 1024, 1, 1, 6200
    %disp([' size(YValidation) = ' num2str(size(YValidation))]); % 1, 1, 1024, 6200

    %keyboard;

    if training == 1
    
        % ------------------ DL Model definition -----------------%
        % E’ la stessa definita dagli autori a pag 16.
        layers = [
            imageInputLayer([size(XTrain,1),1,1],'Name','input')
            % An image input layer inputs 2-D images to a neural network and applies data normalization.
            % Data in this layout has the data format "SSCB" (spatial, spatial, channel, batch).
            % MATLAB normalizza per ciascun canale (asse 3), calcolando la media e deviazione standard sui dati lungo l’asse 4 (batch).
            % Poichè il numero di canali è 1, c'è un unico valore di mean per tutto il tensore di ingresso.
            
            % Inputs:
            
            % Normalization — Data normalization: Data normalization to apply every time data is forward propagated 
            % through the input layer, specified as one of the following:
            %   "zerocenter" (default) — Subtract the mean specified by Mean PER CANALE!
            
            % NormalizationDimension — Normalization dimension
            %   "auto" (default) | "channel" | "element" | "all" 
            %   --> channel-wise
            
            % Mean — Mean for zero-center and z-score normalization
            %   [] (default) | 3-D array | numeric scalar 
            %   --> If Mean is [], then the software automatically sets the property at training or initialization time:
            %       The trainnet function calculates the mean using the training data and uses the resulting value.
            
            % StandardDeviation — Standard deviation for z-score normalization
            %   [] (default) | 3-D array | numeric scalar
            %   --> If StandardDeviation is [], then the software automatically sets the property at training or initialization time:
            %       The trainnet function calculates the standard deviation using the training data and uses the resulting value. 
            
            % Min — Minimum value for rescaling
            %   [] (default) | 3-D array | numeric scalar
            %   --> To specify the Min property, the Normalization must be "rescale-symmetric" or "rescale-zero-one".
            
            % Max — Maximum value for rescaling
            %   [] (default) | 3-D array | numeric scalar
            %   --> To specify the Min property, the Normalization must be "rescale-symmetric" or "rescale-zero-one".
            
            % SplitComplexInputs — Flag to split input data into real and imaginary components
            %   0 (false) (default) | 1 (true)
            
            % Name — Layer name
            %   "" (default) | character vector | string scalar

            % Il parametro 'Normalization' è di default impostato a "zerocenter", che significa che ogni canale dell'immagine in input
            % viene normalizzato sottraendo la media dei pixel. Non viene effettuata una divisione per la deviazione standard, 
            % quindi non si tratta di una normalizzazione standard score (Z-score) ma solo di una centralizzazione rispetto alla media.


            fullyConnectedLayer(size(YTrain,3),'Name','Fully1')
            reluLayer('Name','relu1')
            dropoutLayer(0.5,'Name','dropout1')

            fullyConnectedLayer(4*size(YTrain,3),'Name','Fully2')
            reluLayer('Name','relu2')
            dropoutLayer(0.5,'Name','dropout2')


            fullyConnectedLayer(4*size(YTrain,3),'Name','Fully3')
            reluLayer('Name','relu3')
            dropoutLayer(0.5,'Name','dropout3')


            fullyConnectedLayer(size(YTrain,3),'Name','Fully4')
            regressionLayer('Name','outReg')];
            % Il layer di regressione utilizza i minimi quadrati (mean squared error, MSE)
            % come funzione di perdita per impostazione predefinita.

        %if Training_Size(dd) < miniBatchSize
        %    validationFrequency = Training_Size(dd);
        %else
        %    validationFrequency = floor(Training_Size(dd)/miniBatchSize);
        %end
        if Training_Size_dd < miniBatchSize
            validationFrequency = Training_Size_dd;
        else
            validationFrequency = floor(Training_Size_dd/miniBatchSize);
        end
        VerboseFrequency = validationFrequency;
        options = trainingOptions('sgdm', ...   
            'MiniBatchSize',miniBatchSize, ...
            'MaxEpochs',20, ...
            'InitialLearnRate',1e-1, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropFactor',0.5, ...
            'LearnRateDropPeriod',3, ...
            'L2Regularization',1e-4,...
            'Shuffle','every-epoch', ...
            'ValidationData',{XValidation,YValidation}, ...
            'ValidationFrequency',validationFrequency, ...
            'Plots','none', ... % 'training-progress'
            'Verbose',1, ...    % 1  
            'ExecutionEnvironment', 'cpu', ...
            'VerboseFrequency',VerboseFrequency);
            % ResetInputNormalization — Opzione per ripristinare la normalizzazione del livello di input
            %   1 (true) (predefinito) | 0 (false)

        % ------------- DL Model Training and Prediction -----------------%
        tic
        disp('Start DL training...')
        %[trainedNet,traininfo] = trainNetwork(XTrain,YTrain,layers,options);    
        [trainedNet,~] = trainNetwork(XTrain,YTrain,layers,options);    
        disp('Done')
        toc;
        elapsedTime = toc; % Misura il tempo trascorso in secondi
        elapsedTimeMinutes = elapsedTime / 60; % Converti il tempo trascorso in minuti
        disp(['Elapsed time is ', num2str(elapsedTimeMinutes), 'minutes']);

        %sfile_DeepMIMO=strcat(filename_trainedNet, '_Training_Size_', num2str(Training_Size(dd)), '.mat');
        %save(sfile_DeepMIMO,'trainedNet','-v7.3');
        
        %keyboard;

    else

        % Carica modello pre-allenato caso completo che copre tutta la grid size
        trainedNet = load(filename_trainedNet).trainedNet;
            
    end


    if load_mat == 1
        disp('Import YPredicted from Python')
        YPredicted = h5read(filename_YPredicted_mat, '/YPredicted_mat');
        YPredicted = YPredicted'; % Transpose perchè per la disposizione dei dati in memoria:
        % HDF5 (e quindi h5py) usa la convenzione row-major (C-style)
        % mentre MATLAB usa column-major (Fortran-style)
        disp('Done')
    else
        tic
        disp('Start DL prediction...')
        YPredicted = predict(trainedNet,XValidation); % Inferenza sul set di validazione usato come test: errore!
        disp('Done')
        toc
    end

    
    trainedNet_scaler = trainedNet.Layers(1).Mean; % Estrae la media del primo layer (imageInputLayer)

    disp(['size(YPredicted) = ' num2str(size(YPredicted))]); % 6200, 1024
    disp(['size(trainedNet_scaler) = ' num2str(size(trainedNet_scaler))]); % vettore 1×1×C --> nel nostro caso è uno scalare 1,1

    % Ogni sample in uscita al modello ha un array i rate pari al codebook size,
    % poi bisogna prendere il migliore.
    
    % --------------------- Achievable Rate --------------------------%                    
    %[~,Indmax_OPT]= max(YValidation,[],3);
    [MaxR_OPT_debug,Indmax_OPT]= max(YValidation,[],3); % Luca
    % returns the maximum element along dimension dim. For example, if A is a matrix, then max(A,[],2) returns 
    % a column vector containing the maximum value of each row.
    % Quindi ritorna un vettore lungo 6200 in cui ci sono gli indici corrispondenti al massimo di ogni validation sample
    % che corrispondono ai valori pari a 1 perchè sono stati normalizzati.
    % MaxR_OPT_debug corrisponde a r bar del paper (pag 14).
    %disp(['size(Indmax_OPT):', num2str(size(Indmax_OPT))]); % 1, 1, 1, 6200
    Indmax_OPT = squeeze(Indmax_OPT); %Upper bound on achievable rates
    %disp(['size(Indmax_OPT):', num2str(size(Indmax_OPT))]); % 6200, 1
    % squeeze(A) restituisce un array con gli stessi elementi dell'array di input A, ma con le dimensioni di lunghezza 1 rimosse
    % Ad esempio, se A è un array 3x1x1x2, squeeze(A) restituisce una matrice 3x2.
    MaxR_OPT = single(zeros(numel(Indmax_OPT),1));

    %[~,Indmax_DL] = maxk(YPredicted,kbeams,2); % originale
    [MaxR_DL_luca,Indmax_DL] = maxk(YPredicted,kbeams,2); % Luca
    % max kbeams=1 from dimension 2, quindi trova il max di ogni riga
    % MaxR_DL_luca corrisponde a r cappuccio del paper, cioè all'uscita della rete (pag 14).
    %disp(['size(Indmax_DL):', num2str(size(Indmax_DL))]); % 6200, 1
    %disp(['size(MaxR_DL_luca):', num2str(size(MaxR_DL_luca))]); % 6200, 1
    MaxR_DL = single(zeros(size(Indmax_DL,1),1)); %True achievable rate indexes (size(MaxR_DL)=1)
    % Poichè da YPredicted viene utilizzato solo l'indice del valore massimo, vuol dire che il modello di DL
    % viene usato solamente per ottenere il codebook corrispondendte al massimo perchè quel rate proxy viene ignorato

    for b=1:size(Indmax_DL,1) % 6200
        % YValidation_un = DL_output_un_reshaped = DL_output_un
        %MaxR_DL(b) = max(squeeze(YValidation_un(1,1,Indmax_DL(b,:),b))); %True achievable rates
        MaxR_DL(b) = squeeze(YValidation_un(1,1,Indmax_DL(b),b)); %True achievable rates (Luca)
        MaxR_OPT(b) = squeeze(YValidation_un(1,1,Indmax_OPT(b),b));

        % Count the number of correct predictions (Luca)
        if MaxR_DL(b) == MaxR_OPT(b)
            %validation_accuracy(dd) = validation_accuracy(dd) + 1;
            validation_accuracy = validation_accuracy + 1;
        end

        % debug
        if b==1 || b==(size(Indmax_DL,1) - 1)
            disp(['size(Indmax_DL(b,:)):', num2str(size(Indmax_DL(b,:)))]); % 1, 1
            disp(['MaxR_DL(b):', num2str(MaxR_DL(b))]);
            disp(['MaxR_OPT(b):', num2str(MaxR_OPT(b))]);
            disp(['MaxR_OPT_debug(b):', num2str(MaxR_OPT_debug(b))]); % sempre = 1
            disp(['MaxR_DL_luca(b):', num2str(MaxR_DL_luca(b))]); % sempre <= 1
        end
    end
    % Questa mean fa la media dei risultati di ogni sample del validation set.
    % (Equivalente a trovare la validation accuracy che poi viene plottata nei problemi di classificazione)
    
    %{
    Rate_DL(dd) = mean(MaxR_DL);
    Rate_OPT(dd) = mean(MaxR_OPT);
    Rate_DL_fake(dd) = mean(MaxR_DL_luca);
    % mean returns the mean of the elements of A along the first array dimension whose size does not equal 1.
    LastValidationRMSE(dd) = traininfo.ValidationRMSE(end);
    validation_accuracy(dd) = validation_accuracy(dd) / size(Indmax_DL,1); % Luca
    disp(['size(MaxR_DL):', num2str(size(MaxR_DL))]); % 6200, 1
    disp(['size(MaxR_OPT):', num2str(size(MaxR_OPT))]); % 6200, 1
    disp(['Rate_OPT(dd):', num2str(Rate_OPT(dd))]); % 1
    disp(['Rate_DL(dd):', num2str(Rate_DL(dd))]); % 1
    disp(['Rate_DL_fake(dd):', num2str(Rate_DL_fake(dd))]); % 1
    disp(['LastValidationRMSE(dd):', num2str(LastValidationRMSE(dd))]);
    disp(['validation_accuracy(dd):', num2str(validation_accuracy(dd))]);
    %}
    Rate_DL = mean(MaxR_DL);
    Rate_OPT = mean(MaxR_OPT);
    Rate_DL_fake = mean(MaxR_DL_luca);
    % mean returns the mean of the elements of A along the first array dimension whose size does not equal 1.
    %LastValidationRMSE = traininfo.ValidationRMSE(end);
    validation_accuracy = validation_accuracy / size(Indmax_DL,1); % Luca
    disp(['size(MaxR_DL):', num2str(size(MaxR_DL))]); % 6200, 1
    disp(['size(MaxR_OPT):', num2str(size(MaxR_OPT))]); % 6200, 1
    disp(['Rate_OPT:', num2str(Rate_OPT)]); % 1
    disp(['Rate_DL:', num2str(Rate_DL)]); % 1
    %disp(['Rate_DL_fake:', num2str(Rate_DL_fake)]); % 1
    %disp(['LastValidationRMSE:', num2str(LastValidationRMSE)]);
    disp(['validation_accuracy:', num2str(validation_accuracy)]);
    disp(' ');

    %keyboard;

    %clear trainedNet traininfo YPredicted
    %clear layers options 
        
    %end


    %%%%% TEMP
    if load_mat == 1
        save(filename_Rate_DL_mat,'Rate_DL','-v7.3');
        save(filename_Rate_OPT_mat,'Rate_OPT','-v7.3');
    end
    %%%%% TEMP

    if save_mat_files == 1
        save(filename_Rate_DL,'Rate_DL','-v7.3');
        save(filename_Rate_OPT,'Rate_OPT','-v7.3');
        if load_mat == 1
            save(filename_Rate_DL_mat,'Rate_DL','-v7.3');
            save(filename_Rate_OPT_mat,'Rate_OPT','-v7.3');
        end

        save(filename_trainedNet_scaler,'trainedNet_scaler','-v7.3');
        save(filename_trainedNet,'trainedNet','-v7.3');
        % https://it.mathworks.com/help/deeplearning/networks-from-external-platforms.html
        %exportONNXNetwork(trainedNet, filename_trainedNet_onnx);
        exportNetworkToTensorFlow(trainedNet,filename_trainedNet_tf)

        save(filename_XTrain,'XTrain','-v7.3');
        save(filename_YTrain,'YTrain','-v7.3');
        save(filename_XValidation,'XValidation','-v7.3');
        save(filename_YValidation,'YValidation','-v7.3');
    end
end

end
