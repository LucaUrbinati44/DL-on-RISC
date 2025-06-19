function []=FigC_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,RandP_all,Validation_Ind,Test_Ind,epochs,plot_mode,Training_Size_number,...
                plot_index,plot_rate,plot_test_only,plot_threshold,threshold,...
                MaxR_OPTt_mat, MaxR_DLt_mat, ...
                MaxR_OPTt_py_test, ...
                MaxR_DLt_py_test_20, ...
                MaxR_DLt_py_test_40, ...
                MaxR_DLt_py_test_60, ...
                MaxR_DLt_py_test_80, ...
                MaxR_DLt_py_test_100)
%% Description:
%
% This is the function called by the main script for ploting Figure 10 
% in the original article mentioned below.

global load_H_files load_Delta_H_max load_DL_dataset load_Rates training save_mat_files load_mat_py;
global seed DeepMIMO_dataset_folder DL_dataset_folder network_folder network_folder_py figure_folder figure_folder_py;

if plot_mode == 1

    if epochs == 20
        MaxR_OPTt = MaxR_OPTt_mat;
        MaxR_DLt = MaxR_DLt_mat;
    else
        disp('Error: only 20 is allowed for matVal')
    end

elseif plot_mode == 2

    MaxR_OPTt = MaxR_OPTt_py_test;
    
    if epochs == 20
        MaxR_DLt = MaxR_DLt_py_test_20;
    elseif epochs == 40
        MaxR_DLt = MaxR_DLt_py_test_40;
    elseif epochs == 60
        MaxR_DLt = MaxR_DLt_py_test_60;
    elseif epochs == 80
        MaxR_DLt = MaxR_DLt_py_test_80;
    elseif epochs == 100
        MaxR_DLt = MaxR_DLt_py_test_100;
    end

end

%Training_Size_number=7; % 30000
Training_Size_dd=Training_Size(Training_Size_number);

for rr=1:1:numel(My_ar)

    disp(['---> Plotting FigC with M:', num2str(My_ar(rr)), ', Training_Size:' num2str(Training_Size_dd), ', threshold: ', num2str(threshold), '...']);

    %filename_User_Location=strcat(DeepMIMO_dataset_folder, 'User_Location', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');

    %filename_XTrain=strcat(DL_dataset_folder, 'XTrain', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
    %filename_YTrain=strcat(DL_dataset_folder, 'YTrain', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
    filename_XValidation=strcat(DL_dataset_folder, 'XValidation', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
    filename_YValidation=strcat(DL_dataset_folder, 'YValidation', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');

    filename_DL_input_reshaped=strcat(DL_dataset_folder, 'DL_input_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
    filename_DL_output_reshaped=strcat(DL_dataset_folder, 'DL_output_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
    %filename_DL_output_un_reshaped=strcat(DL_dataset_folder, 'DL_output_un_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
    filename_DL_output_un_complete_reshaped=strcat(DL_dataset_folder, 'DL_output_un_complete_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');
    %filename_trainedNet=strcat(network_folder, 'trainedNet', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
    filename_YPredictedFig7=strcat(network_folder, 'YPredictedFig7', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '.mat');

    if plot_mode == 1
        filename_FigIdxTest=strcat(figure_folder_py, 'FigIdx_mat', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '.png');
        %filename_FigIdxTh=strcat(figure_folder, 'FigIdxTh', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_th', num2str(threshold), '.png');
        filename_FigIdxTestOnly=strcat(figure_folder_py, 'FigIdxTest_matonly', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '.png');
        %filename_FigIdxThTest=strcat(figure_folder, 'FigIdxThTest', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_th', num2str(threshold), '.png');
        
        filename_figRateTest=strcat(figure_folder_py, 'FigRate_mat', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '.png');
        filename_figRateTestTh=strcat(figure_folder_py, 'FigRateTh_mat', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '_th', num2str(threshold), '.png');
        filename_figRateTestOnly=strcat(figure_folder_py, 'FigRateTest_matonly', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '.png');
        filename_figRateThTestOnly=strcat(figure_folder_py, 'FigRateThTest_matonly', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '_th', num2str(threshold), '.png');
    elseif plot_mode == 2
        filename_FigIdxTest=strcat(figure_folder_py, 'FigIdx_py_test', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '.png');
        filename_FigIdxTestOnly=strcat(figure_folder_py, 'FigIdxTest_py_testonly', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '.png');
        
        filename_figRateTest=strcat(figure_folder_py, 'FigRate_py_test', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '.png');
        filename_figRateTestTh=strcat(figure_folder_py, 'FigRateTh_py_test', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '_th', num2str(threshold), '.png');
        filename_figRateTestOnly=strcat(figure_folder_py, 'FigRateTest_py_testonly', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '.png');
        filename_figRateThTestOnly=strcat(figure_folder_py, 'FigRateThTest_py_testonly', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '_th', num2str(threshold), '.png');
    end
    
    %%

    x_plot = 1:(Ur_rows(2)-1000);
    y_plot = 1:181;
    
    if plot_test_only == 1
        if plot_mode == 1
            X = load(filename_XValidation);
            Y = load(filename_YValidation);
        elseif plot_mode == 2

            Training_Ind = RandP_all(1:Training_Size_dd); % Take some user indexes for validation from the end of RandP_all

            load(filename_DL_input_reshaped);
            load(filename_DL_output_reshaped);

            XTrain = single(DL_input_reshaped(:,1,1,Training_Ind));
            disp(['size(XTrain):', size(XTrain)])

            % Calcolo dei parametri di normalizzazione dal dataset di training
            trainedNet_scaler = mean(mean(XTrain, 2)); % Calcola la media su tutte le righe e colonne
            disp(['trainedNet_scaler: ', num2str(trainedNet_scaler)]);
            
            mean_array = repmat(trainedNet_scaler, 1, size(XTrain, 2)); % Crea un array con la media ripetuta
            variance_array = ones(1, size(XTrain, 2)); % Crea un array di varianze uguali a 1
            
            disp(size(DL_input_reshaped))
            X = single(DL_input_reshaped(:,1,1,Test_Ind)); % caricare il test set
            X = (X - mean_array) ./ sqrt(variance_array);
            Y = single(DL_output_reshaped(1,1,:,Test_Ind)); % caricare il test set 
            %disp(size(X))
            %disp(size(Y))
            
            X = reshape(X, Training_Size_dd, 1024);
            %disp(size(X))
            Y = reshape(Y, Training_Size_dd, 1024);
            %disp(size(Y))
            
        end
    else
        if plot_mode == 1
            % Concatena XTrain + XValidation, cioè utilizza DL_input_reshaped
            load(filename_DL_input_reshaped);
            X = DL_input_reshaped;
            disp(['size(X):', num2str(size(X))])
            
            % Concatena YTrain + YValidation, cioè utilizza DL_output_reshaped
            load(filename_DL_output_reshaped);
            Y = DL_output_reshaped;
            disp(['size(Y):', num2str(size(Y))])
        elseif plot_mode == 2
            load(filename_DL_input_reshaped);
            
            Training_Ind = RandP_all(1:Training_Size_dd); % Take some user indexes for validation from the end of RandP_all
            XTrain = single(DL_input_reshaped(:,1,1,Training_Ind));
            %disp(size(XTrain)) % 1024, 1, 1, 10000

            % Calcolo dei parametri di normalizzazione dal dataset di training
            trainedNet_scaler = mean(XTrain, "all"); % Calcola la media su tutte le righe e colonne
            disp(['trainedNet_scaler: ', num2str(trainedNet_scaler)]);
            disp(['size(trainedNet_scaler):', num2str(size(trainedNet_scaler))]); % 1, 1
            
            mean_array = single(repmat(trainedNet_scaler, size(XTrain, 1), 1)); % Crea un array con la media ripetuta
            disp(['size(mean_array):', num2str(size(mean_array))]) % 1024, 1
            variance_array = single(repmat(1, size(XTrain, 1), 1)); % Crea un array di varianze uguali a 1
            disp(['size(variance_array):', num2str(size(variance_array))]) % 1024, 1

            X = DL_input_reshaped;
            disp(['size(X):', num2str(size(X))]) % 1024, 1, 1, 36200
            X = single((X - mean_array) ./ sqrt(variance_array));
            %disp(size(X)) % 1024, 1, 1, 36200
            X = reshape(X, 36200, 1024);
            %disp(size(X))
            
            load(filename_DL_output_reshaped);
            Y = DL_output_reshaped;
            disp(['size(Y):', num2str(size(Y))])
            Y = reshape(Y, 36200, 1024);
            %disp(size(Y))
        end
        
    end


    % Carica modello pre-allenato
    if plot_mode == 1
        filename_trainedNet = strcat(network_folder, 'trainedNet', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '.mat');
        trainedNet = load(filename_trainedNet).trainedNet;

        [~,Indmax_OPT] = max(Y,[],3);

        tic
        YPredictedC = predict(trainedNet,X); % Inferenza sul set di validazione usato come test: errore!
        % Mini-batch size = 128 (default)
        toc
    elseif plot_mode == 2
        %filename_trainedNet = strcat(network_folder_py, 'saved_models/model_py_test', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '.onnx');

        % Parametri
        %model_path_keras = strcat(network_folder_py, 'saved_models_keras/model_py_test', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '.keras');
        model_path_onnx = strcat(network_folder_py, 'saved_models_onnx/model_py_test', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '.onnx');
        %modelfolder_path_tensorflow = strcat(network_folder_py, 'saved_models_onnx/model_py_test', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs));
        %model_path_keras_h5 = strcat(network_folder_py, 'saved_models_onnx/model_py_test', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '.h5');

        % Comando di conversione tf2onnx
        %%cmd = sprintf(['python -m tf2onnx.convert ', '--keras ', keras_model_path, '--output ', onnx_model_path, ' --opset 13']);
        %cmd = sprintf(['python -m tf2onnx.convert ', '--keras ', model_path_keras, '--output ', model_path_onnx, ' --opset 13']);
        % Esegui il comando
        %status = system(cmd);
        %if status ~= 0
        %    error('Errore nella conversione del modello Keras in ONNX.');
        %end

        trainedNet = importNetworkFromONNX(model_path_onnx);
        % Se ottieni errori relativi alla funzione di perdita (mse_custom), puoi ignorarla in MATLAB: 
        % la funzione di perdita è usata solo in fase di training (e il modello è tipicamente esportato per l'inferenza).
        disp(['size(X):', num2str(size(X))]) % 36200, 1024
        %model_py (None, 1024)
        %X2 = zeros(1024, 1, 1, 1, 'single');  % un esempio, non batch intero
        %X2 = zeros(1024, 1, 'single');  % un esempio, non batch intero
        %dlX = dlarray(X2,'CB');
        %X2 = zeros(size(X,2),size(X,1), 'single');  % un esempio, non batch intero
        %dlX = dlarray(X2,'BC');
        %dlX = dlarray(X,'BC');
        dlX = dlarray(X','CB');
        % "S" — Spatial, "C" — Channel, "B" — Batch, "T" — Time, "U" — Unspecified
        trainedNet = initialize(trainedNet, dlX);

        %trainedNet = importNetworkFromTensorFlow(modelfolder_path_tensorflow);
        %trainedNet = importKerasNetwork(model_path_keras_h5) 
        %trainedNet = importKerasNetwork(model_path_keras) % .keras not supported

        %%analyzeNetwork(trainedNet);
        summary(trainedNet)
        %layers = layerGraph(trainedNet);
        %disp(layers)
        disp(trainedNet.Layers)

        [~,Indmax_OPT] = max(Y,[],2);

        tic
        YPredictedC = predict(trainedNet,dlX); % Inferenza sul set di validazione usato come test: errore!
        % Mini-batch size = 128 (default)
        toc

        YPredictedC = single(extractdata(YPredictedC));  % da dlarray a single

        disp(' ')
        disp('1. Recupera layer e connessioni dal dlnetwork')
        lg = layerGraph(trainedNet);
        %disp(lg.Layers)

        disp(' ')
        disp('2. Rimuovi eventuale layer "input" già presente')
        %if any(strcmp({lg.Layers.Name}, 'input'))
        lg = removeLayers(lg, 'input');
        %end
        %disp(lg.Layers)

        disp(' ')
        disp('3. Rimuovi tutte le connessioni esistenti')
        lg = disconnectLayers(lg, 'MatMul_To_AddLayer1011', 'outputOutput');
        %disp(lg.Layers)

        disp(' ')
        disp('4. Aggiungi il input layer coerente con le dimensioni e formati della rete')
        inputLayer1 = featureInputLayer(1024, 'Name', 'input1');
        inputLayer2 = featureInputLayer(1, 'Name', 'input2');
        lg = addLayers(lg, inputLayer1);
        lg = addLayers(lg, inputLayer2);
        %disp(lg.Layers)

        disp(' ')
        disp('5. Connetti l’input al custom layer e output del custom layer all output')
        lg = connectLayers(lg, 'input1', 'MatMul_To_AddLayer1011/in1');
        lg = connectLayers(lg, 'input2', 'MatMul_To_AddLayer1011/in2');
        lg = connectLayers(lg, 'MatMul_To_AddLayer1011', 'outputOutput');
        %disp(lg.Layers)

        disp(' ')
        disp('6. Ricostruisci la rete')
        trainedNetFixed = dlnetwork(lg);

        %disp(' ')
        %disp('Ora puoi esportare in ONNX')
        %model_path_onnx = strcat(network_folder_py, 'saved_models_onnx/model_onnx_mat', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar(rr)), num2str(Mz_ar(rr)), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '.onnx');
        %exportONNXNetwork(trainedNetFixed, model_path_onnx);

        dummyInput = dlarray(zeros(1024,1,'single'), 'CB');
        trainedNetFixed = initialize(dlX, dummyInput);

        tic
        YPredictedCFixed = predict(trainedNetFixed,dlX, dummyInput); % Inferenza sul set di validazione usato come test: errore!
        % Mini-batch size = 128 (default)
        toc

        YPredictedCFixed = single(extractdata(YPredictedCFixed));  % da dlarray a single

    end

    %disp(filename_DL_input_reshaped)
    %disp(size(DL_input_reshaped))
    %disp(size(DL_output_reshaped))
    %disp(size(X))
    %disp(size(Y))

    % Recupera gli indici dei codebook
    % Come mai ho usato DL_output_reshaped invece di YValidation per ottenere Indmax_OPT in Fig7?
    % Perchè dovevo plottare tutti gli utenti nella griglia e DL_output_reshaped li contiene tutti,
    % mentre YValidation ne contiene un sottoinsieme.
    %[~,Indmax_OPT] = max(Y,[],3);
    %disp(['size(Indmax_OPT):', num2str(size(Indmax_OPT))]); % 1, 1, 1, 6200
    Indmax_OPT = squeeze(Indmax_OPT);
    disp(['size(Indmax_OPT):', num2str(size(Indmax_OPT))]); % 6200, 1
    disp(['min(Indmax_OPT):', num2str(min(Indmax_OPT))'])
    %disp(['max(Indmax_OPT):', num2str(max(Indmax_OPT))'])
    disp(max(Indmax_OPT))
    Indmax_OPT = Indmax_OPT.'; % 1, 6200


    %tic
    %YPredictedC = predict(trainedNet,X); % Inferenza sul set di validazione usato come test: errore!
    %% Mini-batch size = 128 (default)
    %toc
    
    %disp(YPredictedC(1,1:10))
    disp(['size(YPredictedC): ', num2str(size(YPredictedC))])

    %[~,Indmax_DL] = maxk(YPredictedC,kbeams,2);
    [~,Indmax_DL] = maxk(YPredictedCFixed,kbeams,2);
    %disp(['size(Indmax_DL):', num2str(size(Indmax_DL))]); % 6200, 1
    %disp(['min(Indmax_DL):', num2str(min(Indmax_DL))'])
    %disp(['max(Indmax_DL):', num2str(max(Indmax_DL))'])
    disp(min(Indmax_DL))
    disp(max(Indmax_DL))



    if plot_test_only == 1

        disp('plot_test_only == 1')

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
        disp(size(ValTest_Ind,1))

        disp('fill sparsity matrix')
        %% Per ogni sample del test set, recupera la sua posizione e il suo rate massimo
        % Se funziona, questo calcolo si può portare dentro a DL_training_4 per semplificare questo script di plot
        values_DL = single(nan(181,200));
        values_OPT = single(nan(181,200));
        for b=1:size(ValTest_Ind,1) % 6200 val o 3100 test
            x = User_Location_norm_ValTest(1,b);
            y = User_Location_norm_ValTest(2,b);

            if plot_index == 1
                values_OPT(x,y) = Indmax_OPT(b);
                values_DL(x,y) = Indmax_DL(b);
            elseif plot_rate == 1
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

    %fC = figure('Name', 'Figure', 'units','pixels', 'Position', [100, 100, 2600, 350]); % typ 800x400
    fC = figure('Name', 'Figure', 'units','pixels', 'Position', [100, 100, 950, 400]); % typ 800x400

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

    %title('(a) Genie Codebook Beams');
    title(['(a) Genie Codebook Beams', '\newline', ' '], 'Interpreter', 'tex');

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
    %if plot_test_only == 1 || plot_threshold == 1
    %    cmap = colormap; % Ottieni la colormap corrente
    %    colormap([1 1 1; cmap]); % Aggiungi il bianco come primo colore
    %    set(gca, 'Color', [1 1 1]); % Imposta il colore di sfondo su bianco
    %else
    %    colormap;
    %end
    %cb = colorbar; % Aggiunge la colorbar e restituisce l'oggetto
    %if plot_threshold == 1 || plot_rate == 1
    %    ylabel(cb, 'Achievable Rate (bps/Hz)','fontsize',11); % Aggiunge una label alla colorbar
    %elseif plot_index == 1
    %    ylabel(cb, 'Codebook index','fontsize',11); % Aggiunge una label alla colorbar
    %end
    %caxis([min_colorbar, max_colorbar]); % Imposta i limiti della scala dei colori    



    %% Subplot 2
    subplot(1, 2, 2); % 1 riga, 2 colonne, secondo subplot

    imagesc([numel(x_plot), 1], [numel(y_plot), 1], values_DL_plot); % values(1,1) in basso a destra

    if plot_mode == 1
        model = 'matVal';
    elseif plot_mode == 2
        model = 'pyTest';
    end
    
    % Creazione del titolo multilinea
    titleStr = ['(b) DL Codebook Beams', newline, '(', model, ' trained with ', num2str(Training_Size_dd), ' for ', num2str(epochs), ' epochs)'];
    % Rimuovi il titolo predefinito
    title('');
    % Ottieni i limiti dell'asse
    xLimits = xlim;
    % Calcola la posizione centrata sopra il subplot
    xCenter = mean(xLimits);
    % Aggiungi il testo centrato
    text(xCenter, -18, titleStr, ...
        'HorizontalAlignment', 'center', ...
        'Interpreter', 'tex', ...
        'FontWeight', 'bold', ...
        'FontSize', 11);

    yticks(y_ticks);
    yticklabels(y_ticks_labels); % The result is {'180', '160', '140', '120', '100', '80', '60', '40', '20', ''}

    if correct_figC == 1    
        xticks(x_ticks_new); % The result is [1, 100, 200]
        xticklabels(x_ticks_labels); % The result is {'1200', '1100', '1000'}

        xlabel('Horizontal user grid direction (y)');
        %ylabel('Vertical user grid direction (x)'); 
    else
        xlabel('Horizontal direction (reversed y-axis)');
        %ylabel('Vertical direction (reversed x-axis)');
    end

    colormap(parula); % Imposta la colormap arcobaleno
    if plot_test_only == 1 || plot_threshold == 1
        cmap = colormap; % Ottieni la colormap corrente
        colormap([1 1 1; cmap]); % Aggiungi il bianco come primo colore
        set(gca, 'Color', [1 1 1]); % Imposta il colore di sfondo su bianco
    else
        colormap;
    end
    %cb = colorbar; % Aggiunge la colorbar e restituisce l'oggetto
    % Aggiungi una colorbar condivisa
    cb = colorbar('Position', [0.92, 0.11, 0.02, 0.7]); % [x, y, width, height]
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
                saveas(fC, filename_figRateThTestOnly);
            else
                saveas(fC, filename_figRateThTest);
            end
        else 
            if plot_test_only == 1
                saveas(fC, filename_figRateTestOnly);
            else
                saveas(fC, filename_figRateTest);
            end
        end
    elseif plot_index == 1
        if plot_test_only == 1
            saveas(fC, filename_FigIdxTestOnly);
        else
            saveas(fC, filename_FigIdxTest);
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