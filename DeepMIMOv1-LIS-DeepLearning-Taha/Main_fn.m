function [Rate_DL,Rate_OPT]=Main_fn(L,My,Mz,M_bar,K_DL,Pt,kbeams,Training_Size)
%% Description:
%
% This is the function called by the main script for ploting Figure 10 
% in the original article mentioned below.
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

disp('-------------------------------------------------------------');

% Luca variables to control the flow
load_mat_files = 1;
load_Delta_H_max = 1;
load_DL_dataset = 1;
load_Rates = 1;

%% System Model Parameters

disp('---> System Model Parameters');

params.scenario='O1_28'; % DeepMIMO Dataset scenario: http://deepmimo.net/

Ut_row = 850; % user Ut row number
Ut_element = 90; % user Ut position from the row chosen above

% ERROR: nel paper è 1000 1300
Ur_rows = [1000 1200]; % user Ur rows

% we select BS 3 in the 'O1' scenario to be the LIS
params.active_BS=3; % active basestation(/s) in the chosen scenario

% Note: The axes of the antennas match the axes of the ray-tracing scenario
Mx = 1;  % number of LIS reflecting elements across the x axis
M = Mx.*My.*Mz; % Total number of LIS reflecting elements 
D_Lambda = 0.5; % Antenna spacing relative to the wavelength
BW = 100e6; % Bandwidth
K = 512; % number of subcarriers
    
%% DeepMIMO Dataset Generation

disp('---> DeepMIMO Dataset Generation');

% Note: The axes of the antennas match the axes of the ray-tracing scenario
params.num_ant_x= Mx;             % Number of the UPA antenna array on the x-axis 
params.num_ant_y= My;             % Number of the UPA antenna array on the y-axis 
params.num_ant_z= Mz;             % Number of the UPA antenna array on the z-axis
params.ant_spacing=D_Lambda;          % ratio of the wavelnegth; for half wavelength enter .5        
params.bandwidth= BW*1e-9;            % The bandiwdth in GHz 
params.num_OFDM= K;                   % Number of OFDM subcarriers
params.OFDM_sampling_factor=1;        % The constructed channels will be calculated only at the sampled subcarriers (to reduce the size of the dataset)
params.OFDM_limit=K_DL*1;         % Only the first params.OFDM_limit subcarriers will be considered when constructing the channels
params.num_paths=L;               % Maximum number of paths to be considered (a value between 1 and 25), e.g., choose 1 if you are only interested in the strongest path
params.saveDataset=0;
%disp([' Calculating for K_DL = ' num2str(K_DL)]);          
%disp([' Calculating for L = ' num2str(params.num_paths)]);

% ------------------ DeepMIMO "Ut" Dataset Generation -----------------%
params.active_user_first=Ut_row; 
params.active_user_last=Ut_row;
params.filename="Ht_"+My+Mz;

if load_mat_files == 1
    disp('Loading '+params.filename+'...');
    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/DeepMIMO_dataset_', params.filename, '.mat');
    load(sfile_DeepMIMO);
    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/params_', params.filename, '.mat');
    load(sfile_DeepMIMO);
    disp('Done');
else
    params.saveDataset=1;
    tic
    disp('Generating '+params.filename+'...');
    DeepMIMO_dataset=DeepMIMO_generator(params);
    disp('Done');
    toc
end

% .channel restituisce la matrice di canale tra l'active base station {1}, che in realtà è la RIS
% e l'utente {Ut_element} che in realtà è la BS trasmettitore
Ht = single(DeepMIMO_dataset{1}.user{Ut_element}.channel);
%disp([' size(DeepMIMO_dataset) = ' num2str(size(DeepMIMO_dataset))]); % returns 1
disp([' size(Ht) = ' num2str(size(Ht))]);
%keyboard; 
clear DeepMIMO_dataset


Ur_rows_step = 100; % access the dataset 100 rows at a time
% indica la posizione del RX lungo asse y, cioè la sua posizione nella riga, quindi la colonna della griglia degli utenti
Ur_rows_grid=Ur_rows(1):Ur_rows_step:Ur_rows(2); % [1000 1100 1200]


% ------------------ DeepMIMO "Delta_H_max" Generation -----------------%

if load_Delta_H_max == 1
    disp('Loading Delta_H_max...');
    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/Delta_H_max_', num2str(My), num2str(Mz), '.mat');
    load(sfile_DeepMIMO);
    disp('Done');
else
    % ------------------ DeepMIMO "Ur" Dataset Generation -----------------%            
    Delta_H_max = single(0); % initilizzato a 0

    %init=1; % for generate specific datasets

    % Divide la griglia in tre regioni verticali e itera su ognuna di esse
    for pp = 1:1:numel(Ur_rows_grid)-1 % 3-1=2, loop for Normalizing H
        clear DeepMIMO_dataset
        params.active_user_first=Ur_rows_grid(pp);
        params.active_user_last=Ur_rows_grid(pp+1)-1; 
        params.filename="Hr_"+pp+"_"+My+Mz;

        if load_mat_files == 1 %&& init == 0
            disp('Loading '+params.filename+'...');
            sfile_DeepMIMO=strcat('./DeepMIMO Dataset/DeepMIMO_dataset_', params.filename, '.mat');
            load(sfile_DeepMIMO);
            sfile_DeepMIMO=strcat('./DeepMIMO Dataset/params_', params.filename, '.mat');
            load(sfile_DeepMIMO);
            disp('Done');
            %init = 1;
        else
            params.saveDataset=1;
            tic
            disp('Generating '+params.filename+'...');
            [DeepMIMO_dataset,params]=DeepMIMO_generator(params);
            disp('Done');
            toc
        end

        % Per ogni griglia verticale, lavora su ogni utente contenuto all'interno
        for u=1:params.num_user
            % .channel restituisce la matrice di canale tra l'active base station {1} e le righe di utenti {u}
            Hr = single(conj(DeepMIMO_dataset{1}.user{u}.channel)); % single precision, conj = complesso coniugato        

            Delta_H = max(max(abs(Ht.*Hr))); % qui è senza rumore (sotto è con rumore)
            if Delta_H >= Delta_H_max
                Delta_H_max = single(Delta_H);
            end    
        end

        disp([' size(Hr) = ' num2str(size(Hr))]);
        %keyboard; 
    end
    clear Delta_H

    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/Delta_H_max_', num2str(My), num2str(Mz), '.mat');
    save(sfile_DeepMIMO,'Delta_H_max','-v7.3'); % save(filename, variables, options),

    disp(' Delta_H_max: ');
    disp(Delta_H_max);
    disp([' size(Delta_H_max) = ' num2str(size(Delta_H_max))]);

end

%% Deep Learning Dataset Generation

disp('---> Deep Learning Dataset Generation');

No_user_pairs = (Ur_rows(2)-Ur_rows(1))*181; % Number of users (= Number of TX-RX pairs) = Total numer of users in Grid 1
disp([' params.num_user = ' num2str(params.num_user)]);
disp([' No_user_pairs = ' num2str(No_user_pairs)]);
% In the 'O1' scenario where every row consists of 181 points.
% Since the number of BS antennas is one, the number of pairs is equal to the number of users.
RandP_all = randperm(No_user_pairs).'; % Random permutation = shuffled user indexes

Validation_Size = 6200; % Validation dataset Size
%Validation part for the actual achievable rate perf eval
Validation_Ind = RandP_all(end-Validation_Size+1:end); % Take some user indexes for validation from the end of RandP_all


if load_DL_dataset == 1
    disp('Loading DL_input and DL_output...');
    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/DL_input_', num2str(My), num2str(Mz), '.mat');
    load(sfile_DeepMIMO);
    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/DL_output_', num2str(My), num2str(Mz), '.mat');
    load(sfile_DeepMIMO);
    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/DL_output_un_', num2str(My), num2str(Mz), '.mat');
    load(sfile_DeepMIMO);
    disp('Done');
else   

    [~,VI_sortind] = sort(Validation_Ind); % contiene gli indici che ordinano Validation_Ind in ordine crescente.
    [~,VI_rev_sortind] = sort(VI_sortind); % 
    % L'idea dietro questi due ordinamenti consecutivi è di ottenere un array di indici (VI_rev_sortind)
    % che può essere utilizzato per riportare un array ordinato alla sua disposizione originale. 

    % BF codebook parameters
    over_sampling_x=1;            % The beamsteering oversampling factor in the x direction
    over_sampling_y=1;            % The beamsteering oversampling factor in the y direction
    over_sampling_z=1;            % The beamsteering oversampling factor in the z direction
    % Generating the BF codebook 
    [BF_codebook]=sqrt(Mx*My*Mz)*UPA_codebook_generator(Mx,My,Mz,over_sampling_x,over_sampling_y,over_sampling_z,D_Lambda);
    codebook_size=size(BF_codebook,2); % (1024, 1024) --> 1024
    disp(['Codebook generated with size ', num2str(codebook_size)]);


    %--- Accounting SNR in each rate calculations
    %--- Defining Noisy channel measurements
    Gt=3;             % dBi
    Gr=3;             % dBi
    NF=5;             % Noise figure at the User equipment
    Process_Gain=10;  % Channel estimation processing gain
    noise_power_dB=-204+10*log10(BW/K)+NF-Process_Gain; % Noise power in dB
    SNR = 10^(.1*(-noise_power_dB)) * (10^(.1*(Gt+Gr+Pt)))^2; % Signal-to-noise ratio
    % La formula 10^(.1 * x) è usata per convertire un valore in dB a una scala lineare.
    % channel estimation noise
    noise_power_bar=10^(.1*(noise_power_dB))/(10^(.1*(Gt+Gr+Pt))); 


    DL_input = single(zeros(M_bar*K_DL*2,No_user_pairs)); % Questo è il dataset: ogni entry è un sample che rappresenta una coppia utente-BS, 8x64x2(real/img)=1024 
    DL_output = single(zeros(No_user_pairs,codebook_size)); % Queste sono le label per ogni sample, cioè il rate (uno per ogni codebook)
    DL_output_un=  single(zeros(numel(Validation_Ind),codebook_size));
    Delta_H_bar_max = single(0);
    count=0;

    % The active channel sensors are randomly selected from the M UPA antennas
    Rand_M_bar_all = randperm(M);
    Rand_M_bar =unique(Rand_M_bar_all(1:M_bar)); % prendi i primi M_bar valori di M dopo aver mischiato i valori di M randomly
    disp([' Calculating for M_bar = ' num2str(M_bar)]);  

    % Keep only the coefficients of Ht that are related to M_bar
    Ht_bar = reshape(Ht(Rand_M_bar,:),M_bar*K_DL,1);

    u_step=100;
    Htx=repmat(Ht(:,1),1,u_step); % Replica Ht array verticale lungo le colonne per u_step volte

    % Per ogni regione verticale
    for pp = 1:1:numel(Ur_rows_grid)-1
        clear DeepMIMO_dataset 
        %disp(['Starting received user access ' num2str(pp)]);
        params.active_user_first=Ur_rows_grid(pp);
        params.active_user_last=Ur_rows_grid(pp+1)-1;
        params.filename="Hr_"+pp+"_"+My+Mz;
        params.saveDataset=0; % Non lo salvo perchè uguale a quello già calcolato in precedenza.
        % Senza ricalcolare il dataset, si potrebbe re-importarlo.
        % Forse occupava troppo spazio e in un pc portatile non ci stava, così hanno deciso di fare clear 
        % e ricalcolarla al bisogno.

        if load_mat_files == 1
            disp('Loading '+params.filename+'...');
            sfile_DeepMIMO=strcat('./DeepMIMO Dataset/DeepMIMO_dataset_', params.filename, '.mat');
            load(sfile_DeepMIMO);
            sfile_DeepMIMO=strcat('./DeepMIMO Dataset/params_', params.filename, '.mat');
            load(sfile_DeepMIMO);
            disp('Done');
        else
            params.saveDataset=1;
            tic
            disp('Generating '+params.filename+'...');
            [DeepMIMO_dataset,params]=DeepMIMO_generator(params);
            disp('Done');
            toc
        end

        Hrx=zeros(M,u_step);
        for u=1:u_step:params.num_user % Per ogni utente dentro una regione verticale a step di 100 (u_step)                   
            for uu=1:1:u_step % A step di 1 dentro i 100 (praticamente potevano fare un unico loop invece di fare u e uu)
                Hr = single(conj(DeepMIMO_dataset{1}.user{u+uu-1}.channel)); % ritorna una matrice?
                %disp([' size(Hr) = ' num2str(size(Hr))]);  % size(Hr) = 1024, 64

                % Keep only the coefficients of Hr that are related to M_bar --> Filtra per righe
                Hr_bar = reshape(Hr(Rand_M_bar,:),M_bar*K_DL,1);
                %disp([' size(Hr_bar) = ' num2str(size(Hr_bar))]);  % size(Hr_bar) = 8x64=512, 1 --> è un array
                
                %--- Constructing the sampled channel
                n1=sqrt(noise_power_bar/2)*(randn(M_bar*K_DL,1)+1j*randn(M_bar*K_DL,1));
                n2=sqrt(noise_power_bar/2)*(randn(M_bar*K_DL,1)+1j*randn(M_bar*K_DL,1));
                H_bar = ((Ht_bar+n1).*(Hr_bar+n2)); % .* = element-wise multiplication = hadamard product
                %disp([' size(H_bar) = ' num2str(size(H_bar))]); % size(H_bar) = 512, 1

                % Eliminazione i dalla parte immaginaria per poter usarla come input dell'algoritmo
                % [ ] fa una concatenazione orizzontale
                % .' resistuisce la trasposta
                % Con il reshape si ottiene un vettore colonna perchè imponiamo una colonna con 1 e gli lasciamo calcolare le righe con []
                % (specify [] for the first dimension to let reshape automatically calculate the appropriate number of rows)
                % Il risultato finale è un vettore colonna che ha prima tutti gli elementi della matrice reale
                % e poi tutti gli elementi della matrice immaginaria.
                % Questo vettore colonna viene appeso verticalmente a DL_input per ogni utente della Griglia 1, un utente alla volta
                % shape(DL_input) = (1024, 36200) perchè 
                % 8 celle attive x 64 subcarriers x 2 (real/img) = 1024
                % numero link utenti RIS = numero totale utenti = (1200 - 1000) x 181 = 36200
                DL_input(:,u+uu-1+((pp-1)*params.num_user))= reshape([real(H_bar) imag(H_bar)].',[],1);
                %disp(['size(DL_input): ' num2str(size(DL_input))]); % size(DL_input) = 1024, 36200
                
                Delta_H_bar = max(max(abs(H_bar))); % Massimo di ogni colonna e poi massimo della riga risultate 
                % --> Si ottiene il valore massimo della matrice 2D.
                if Delta_H_bar >= Delta_H_bar_max
                    Delta_H_bar_max = single(Delta_H_bar);
                end
                Hrx(:,uu)=Hr(:,1); % Perchè tiene solo la prima colonna?

            end

            %--- Actual achievable rate for performance evaluation
            % Da qui al clear non ho capito cosa succede ???
            H = Htx.*Hrx;
            H_BF=H.'*BF_codebook;
            SNR_sqrt_var = abs(H_BF);
            for uu=1:1:u_step
                if sum((Validation_Ind == u+uu-1+((pp-1)*params.num_user)))
                    count=count+1;
                    DL_output_un(count,:) = single( sum( log2( 1+( SNR*( (SNR_sqrt_var(uu,:)).^2 ) ) ), 1) ); 
                    % returns the sum along dimension dim. For example, if A is a matrix, 
                    % then sum(A,2) returns a column vector containing the sum of each row
                end
            end
            %--- Label for the sampled channel
            R = single(log2(1+(SNR_sqrt_var/Delta_H_max).^2)); % size(R) = 100, 1024
            %disp([' size(R):', num2str(size(R))]);

            % --- DL output normalization
            Delta_Out_max = max(R,[],2);
            % returns the maximum element along dimension dim, i.e., 
            % returns a column vector containing the maximum value of each row.
            if ~sum(Delta_Out_max == 0) % Se non ci sono elementi nulli
            % Normalizza la diagonale. Perchè solo la diagonale? A cosa mi serve normalizzare il rate?
            Rn=diag(1./Delta_Out_max)*R; % Matrice diagonale con array sulla diagonale
            end
            DL_output(u+((pp-1)*params.num_user):u+((pp-1)*params.num_user)+u_step-1,:) = Rn; % size(DL_output) = 32600, 1024

            %disp([' size(DL_output) = ' num2str(size(DL_output))]);

        end

        %keyboard; 
    end
    clear u Delta_H_bar R Rn

    %-- Sorting back the DL_output_un
    DL_output_un = DL_output_un(VI_rev_sortind,:);

    %--- DL input normalization (questo l'ho capito)
    DL_input= 1*(DL_input/Delta_H_bar_max); %%%%% Normalized from -1->1 %%%%%
    disp([' Delta_H_bar_max:', num2str(Delta_H_bar_max)]);
    disp([' size(Delta_H_bar_max):', num2str(size(Delta_H_bar_max))]);

    %disp([' size(DL_input) = ' num2str(size(DL_input))]);
    %disp([' size(DL_output) = ' num2str(size(DL_output))]);

    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/DL_input_', num2str(My), num2str(Mz), '.mat');
    save(sfile_DeepMIMO,'DL_input','-v7.3');
    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/DL_output_', num2str(My), num2str(Mz), '.mat');
    save(sfile_DeepMIMO,'DL_output','-v7.3');
    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/DL_output_un_', num2str(My), num2str(Mz), '.mat');
    save(sfile_DeepMIMO,'DL_output_un','-v7.3');

end

%keyboard;

%% DL Beamforming

disp('---> DL Beamforming');

if load_Rates == 1
    disp('Loading trainedNet, traininfo, Rate_DL, Rate_OPT...');
    %sfile_DeepMIMO=strcat('./DeepMIMO Dataset/trainedNet.mat');
    %load(sfile_DeepMIMO);
    %sfile_DeepMIMO=strcat('./DeepMIMO Dataset/traininfo.mat');
    %load(sfile_DeepMIMO);
    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/Rate_DL_', num2str(My), num2str(Mz), '.mat');
    load(sfile_DeepMIMO);
    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/Rate_OPT_', num2str(My), num2str(Mz), '.mat');
    load(sfile_DeepMIMO);
    disp('Done');
else

    miniBatchSize  = 500; % Size of the minibatch for the Deep Learning

    % Preallocation of output variables
    Rate_DL = zeros(1,length(Training_Size)); 
    Rate_OPT = Rate_DL;
    LastValidationRMSE = Rate_DL;

    % ------------------ Training and Testing Datasets -----------------%
    DL_output_reshaped = reshape(DL_output.',1,1,size(DL_output,2),size(DL_output,1));
    DL_output_reshaped_un = reshape(DL_output_un.',1,1,size(DL_output_un,2),size(DL_output_un,1));
    DL_input_reshaped= reshape(DL_input,size(DL_input,1),1,1,size(DL_input,2));

    % Per ogni punto del grafico, viene allenato un modello
    for dd=1:1:numel(Training_Size)
        disp([' Calculating for Dataset Size = ' num2str(Training_Size(dd))]);
        % Get a random number of indeces equal to the content of one element of the Training_Size array
        Training_Ind   = RandP_all(1:Training_Size(dd));

        % Use the indexes to extract the actual sampled used for training from DL_input.
        % This is why DL_input is designed to be No_user_pairs log while in reality it is shorter and equal to Training_Ind = Training_Size(dd).
        XTrain = single(DL_input_reshaped(:,1,1,Training_Ind));
        YTrain = single(DL_output_reshaped(1,1,:,Training_Ind));
        XValidation = single(DL_input_reshaped(:,1,1,Validation_Ind));
        YValidation = single(DL_output_reshaped(1,1,:,Validation_Ind));
        YValidation_un = single(DL_output_reshaped_un);

        disp([' size(XTrain) = ' num2str(size(XTrain))]);
        disp([' size(YTrain) = ' num2str(size(YTrain))]);
        disp([' size(XValidation) = ' num2str(size(XTrain))]);
        disp([' size(YValidation) = ' num2str(size(YTrain))]);
        
        % ------------------ DL Model definition -----------------%
        layers = [
            imageInputLayer([size(XTrain,1),1,1],'Name','input')

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

        if Training_Size(dd) < miniBatchSize
            validationFrequency = Training_Size(dd);
        else
            validationFrequency = floor(Training_Size(dd)/miniBatchSize);
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
            'Verbose',0, ...    % 1  
            'ExecutionEnvironment', 'cpu', ...
            'VerboseFrequency',VerboseFrequency);
        
        % ------------- Genie-Aided Reflection Beamforming -----------------%
        [~,Indmax_OPT]= max(YValidation,[],3); % returns the maximum element along dimension dim. 
        % For example, if A is a matrix, then max(A,[],2) returns a column vector containing the maximum value of each row.
        % Quindi ritorna un vettore
        Indmax_OPT = squeeze(Indmax_OPT); %Upper bound on achievable rates
        % squeeze(A) restituisce un array con gli stessi elementi dell'array di input A, ma con le dimensioni di lunghezza 1 rimosse
        MaxR_OPT = single(zeros(numel(Indmax_OPT),1));   

        % ------------- DL Model Training and Prediction -----------------%
        tic
        disp('Start DL training...')
        [trainedNet,traininfo]  = trainNetwork(XTrain,YTrain,layers,options);    
        disp('Done')
        toc;
        elapsedTime = toc; % Misura il tempo trascorso in secondi
        elapsedTimeMinutes = elapsedTime / 60; % Converti il tempo trascorso in minuti
        disp(['Elapsed time is ', num2str(elapsedTimeMinutes), 'minutes']);
        
        %keyboard;

        tic
        disp('Start DL testing...')
        YPredicted = predict(trainedNet,XValidation); % Inferenza sul set di validazione usato come test: errore!
        disp('Done')
        toc

        disp(['size(YPredicted) = ' num2str(size(YPredicted))]);
        
        % --------------------- Achievable Rate --------------------------%                    
        [~,Indmax_DL] = maxk(YPredicted,kbeams,2); % max kbeams (=1) from dimension 2
        MaxR_DL = single(zeros(size(Indmax_DL,1),1)); %True achievable rate indexes (size(MaxR_DL)=1)
        disp([' size(MaxR_DL) = ' num2str(size(MaxR_DL))]);
        for b=1:size(Indmax_DL,1) % Recupera da DL_output il rate usando l'indice del massimo della predizione
            % Ma quindi: cos'è YPredicted?
            % YValidation_un = DL_output_reshaped_un = DL_output_un
            MaxR_DL(b) = max(squeeze(YValidation_un(1,1,Indmax_DL(b,:),b))); %True achievable rates
            MaxR_OPT(b) = squeeze(YValidation_un(1,1,Indmax_OPT(b),b));
        end
        Rate_DL(dd) = mean(MaxR_DL); % Fa la media nel caso in cui kbeams > 1
        Rate_OPT(dd) = mean(MaxR_OPT);
        LastValidationRMSE(dd) = traininfo.ValidationRMSE(end);                                          
        disp(['Rate_DL(dd):', num2str(Rate_DL(dd))]);
        disp(['Rate_OPT(dd):', num2str(Rate_OPT(dd))]);
        disp(['LastValidationRMSE(dd):', num2str(LastValidationRMSE(dd))]);
        disp(' ');

        sfile_DeepMIMO=strcat('./DeepMIMO Dataset/trainedNet_', num2str(My), num2str(Mz), '_', num2str(dd), '.mat');
        save(sfile_DeepMIMO,'trainedNet','-v7.3');
        sfile_DeepMIMO=strcat('./DeepMIMO Dataset/traininfo_', num2str(My), num2str(Mz), '_', num2str(dd), '.mat');
        save(sfile_DeepMIMO,'traininfo','-v7.3');

        %keyboard;

        clear trainedNet traininfo YPredicted
        clear layers options Rate_DL_Temp MaxR_DL_Temp Highest_Rate
        
    end

    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/Rate_DL_', num2str(My), num2str(Mz), '.mat');
    save(sfile_DeepMIMO,'Rate_DL','-v7.3');
    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/Rate_OPT_', num2str(My), num2str(Mz), '.mat');
    save(sfile_DeepMIMO,'Rate_OPT','-v7.3');
end

%% End of script

disp('--> End of script. If you continue, you will lose access to all variables');

keyboard;

end
