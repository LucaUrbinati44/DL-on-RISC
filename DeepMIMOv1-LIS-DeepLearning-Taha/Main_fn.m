function [Rate_DL,Rate_OPT]=Main_fn(output_folder,seed,L,My,Mz,M_bar,K_DL,Pt,kbeams,Ur_rows,Training_Size)
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
save_mat_files = 0;
plot_fig7 = 1;

%% System Model Parameters

disp('---> System Model Parameters');

params.scenario='O1_28'; % DeepMIMO Dataset scenario: http://deepmimo.net/

Ut_row = 850; % user Ut row number
Ut_element = 90; % user Ut position from the row chosen above

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
filename_Ht=strcat('Ht', '_seed', '_grid', num2str(Ur_rows(2)), num2str(seed), '_M', num2str(My), num2str(Mz));
filename_Hr=strcat('Hr', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz));
filename_Delta_H_max=strcat(output_folder, 'DeepMIMO Dataset/Delta_H_max', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '.mat');
filename_DL_input=strcat(output_folder, 'DeepMIMO Dataset/DL_input', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '.mat');
filename_User_Location=strcat(output_folder, 'DeepMIMO Dataset/User_Location', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '.mat');
filename_params=strcat(output_folder, 'DeepMIMO Dataset/params', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '.mat');
filename_DL_output=strcat(output_folder, 'DeepMIMO Dataset/DL_output', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '.mat');
filename_DL_output_un=strcat(output_folder, 'DeepMIMO Dataset/DL_output_un', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '.mat');
filename_DL_input_reshaped=strcat(output_folder, 'DeepMIMO Dataset/DL_input_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '.mat');
filename_DL_output_reshaped=strcat(output_folder, 'DeepMIMO Dataset/DL_output_reshaped', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '.mat');
filename_trainedNet=strcat(output_folder, 'DeepMIMO Dataset/trainedNet', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar));
filename_Rate_DL=strcat(output_folder, 'DeepMIMO Dataset/Rate_DL', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '.mat');
filename_Rate_OPT=strcat(output_folder, 'DeepMIMO Dataset/Rate_OPT', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '.mat');
filename_YPredictedFig7=strcat(output_folder, 'DeepMIMO Dataset/YPredictedFig7', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '_Mbar', num2str(M_bar), '.mat');

% ------------------ DeepMIMO "Ut" Dataset Generation -----------------%
params.active_user_first=Ut_row; 
params.active_user_last=Ut_row;

if load_mat_files == 1
    disp(['Loading ', filename_Ht, '...']);
    sfile_DeepMIMO=strcat(output_folder, 'DeepMIMO Dataset/DeepMIMO_dataset_', filename_Ht, '.mat');
    load(sfile_DeepMIMO);
    sfile_DeepMIMO=strcat(output_folder, 'DeepMIMO Dataset/params_', filename_Ht, '.mat');
    load(sfile_DeepMIMO);
    disp('Done');
else
    params.saveDataset=1;
    tic
    params.filename = filename_Ht;
    disp(['Generating ', filename_Ht, '...']);
    DeepMIMO_dataset=DeepMIMO_generator(params);
    disp('Done');
    toc
end

% .channel restituisce la matrice di canale tra l'active base station {1}, che in realtà è la RIS
% e l'utente {Ut_element} che in realtà è la BS trasmettitore
Ht = single(DeepMIMO_dataset{1}.user{Ut_element}.channel);
%disp([' size(DeepMIMO_dataset) = ' num2str(size(DeepMIMO_dataset))]); % returns 1
disp([' size(Ht) = ' num2str(size(Ht))]); % size(Ht) = 1024, 64
%keyboard; 
clear DeepMIMO_dataset


Ur_rows_step = 100; % access the dataset 100 rows at a time
% indica la posizione del RX lungo asse y, cioè la sua posizione nella riga, quindi la colonna della griglia degli utenti
Ur_rows_grid=Ur_rows(1):Ur_rows_step:Ur_rows(2); % [1000 1100 1200] --> [1000 1100 1200 1300] 


% ------------------ DeepMIMO "Delta_H_max" Generation -----------------%

if load_Delta_H_max == 1
    disp('Loading Delta_H_max...');
    load(filename_Delta_H_max);
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
        filename_Hrpp=strcat(filename_Hr, '_pp', num2str(pp));

        if load_mat_files == 1 %&& init == 0
            disp(['Loading ', filename_Hrpp, '...']);
            sfile_DeepMIMO=strcat(output_folder, 'DeepMIMO Dataset/DeepMIMO_dataset_', filename_Hrpp, '.mat');
            load(sfile_DeepMIMO);
            sfile_DeepMIMO=strcat(output_folder, 'DeepMIMO Dataset/params_', filename_Hrpp, '.mat');
            load(sfile_DeepMIMO);
            disp('Done');
            %init = 1;
        else
            params.saveDataset=1;
            tic
            params.filename = filename_Hrpp;
            disp(['Generating ', filename_Hrpp, '...']);
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
                Delta_H_max = single(Delta_H); % E' il massimo della matrice H senza rumore
            end    
        end

        disp([' size(Hr) = ' num2str(size(Hr))]);
        %keyboard; 
    end
    clear Delta_H

    if save_mat_files == 1
        save(filename_Delta_H_max,'Delta_H_max','-v7.3'); % save(filename, variables, options),
    end

    disp(' Delta_H_max: ');
    disp(Delta_H_max);
    disp([' size(Delta_H_max) = ' num2str(size(Delta_H_max))]);

end

%keyboard; % Pause

%% Deep Learning Dataset Generation = Genie-Aided Reflection Beamforming

disp('---> Deep Learning Dataset Generation');

No_user_pairs = (Ur_rows(2)-Ur_rows(1))*181; % Number of users (= Number of TX-RX pairs) = Total numer of users in Grid 1
disp([' No_user_pairs = ' num2str(No_user_pairs)]);
% In the 'O1' scenario where every row consists of 181 points.
% Since the number of BS antennas is one, the number of pairs is equal to the number of users.
RandP_all = randperm(No_user_pairs).'; 
% randperm(n) restituisce un vettore contenente una permutazione casuale di numeri interi da 1 a n, senza elementi ripetuti.
% Serve per ottenere gli indici mischiati (shuffled)

Validation_Size = 6200; % Validation dataset Size
%Validation part for the actual achievable rate perf eval
Validation_Ind = RandP_all(end-Validation_Size+1:end); % Take some user indexes for validation from the end of RandP_all

% Calcola indici di Validation_Ind quando ordinato in ordine crescente
[~,VI_sortind] = sort(Validation_Ind);
% Calcola indici di VI_sortind, cioè degli indici precedenti, quando ordinati in ordine crescente
[~,VI_rev_sortind] = sort(VI_sortind); 
% L'idea dietro questi due ordinamenti consecutivi è di ottenere un array di indici (VI_rev_sortind)
% che può essere utilizzato per riportare un array ordinato (l'output ~ del primo sort) 
% alla sua disposizione originale (Validation_Ind). 

u_step=100;

%keyboard; % Pause

if load_DL_dataset == 1
    disp(['Loading DL_input and DL_output...']);
    load(filename_DL_input);
    load(filename_DL_output);
    load(filename_DL_output_un);
    disp('Done');
else   

    % BF codebook parameters
    over_sampling_x=1;            % The beamsteering oversampling factor in the x direction
    over_sampling_y=1;            % The beamsteering oversampling factor in the y direction
    over_sampling_z=1;            % The beamsteering oversampling factor in the z direction
    % Generating the BF codebook 
    [BF_codebook]=sqrt(Mx*My*Mz)*UPA_codebook_generator(Mx,My,Mz,over_sampling_x,over_sampling_y,over_sampling_z,D_Lambda); % (1024, 1024)
    codebook_size=size(BF_codebook,2);
    % significa che:
    %  - ogni riga è un possibile codebook, cioè 1024 valori di fasi per i 32x32=1024 elementi della RIS
    %  - l'insieme delle righe sono tutti i possibili codebook. Sono anch'essi 1024 perchè sono stati ottenuti dalla DFT
    % che implica che essi siano ortogonali tra loro (se ne moltiplico uno per il suo complesso coniugato ottengo 0)
    disp([' size(BF_codebook)', num2str(size(BF_codebook))]); % Luca
    disp([' Codebook size: ', num2str(codebook_size)]); % 1024


    %--- Accounting SNR in each rate calculations
    %--- Defining Noisy channel measurements
    Gt=3;             % dBi
    Gr=3;             % dBi
    NF=5;             % Noise figure at the User equipment
    Process_Gain=10;  % Channel estimation processing gain
    noise_power_dB=-204+10*log10(BW/K)+NF-Process_Gain; % Noise power in dB
    SNR = 10^(.1*(-noise_power_dB)) * (10^(.1*(Gt+Gr+Pt)))^2; % Signal-to-noise ratio: Pt / Noise 
    % Formula diversa da quella del paper pag 44307. Come mai?
    % La formula 10^(.1 * x) è usata per convertire un valore in dB a una scala lineare.
    % channel estimation noise
    noise_power_bar=10^(.1*(noise_power_dB))/(10^(.1*(Gt+Gr+Pt))); 


    DL_input = single(zeros(M_bar*K_DL*2,No_user_pairs)); % Questo è il dataset: ogni entry è un sample che rappresenta una coppia utente-BS, 8x64x2(real/img)=1024 
    User_Location = single(zeros(3,No_user_pairs)); % x, y, z location of users
    DL_output = single(zeros(No_user_pairs,codebook_size)); % Queste sono le label per ogni sample, cioè il rate (uno per ogni codebook)
    DL_output_un =  single(zeros(numel(Validation_Ind),codebook_size));
    Delta_H_bar_max = single(0);
    count=0;

    % The active channel sensors are randomly selected from the M UPA antennas
    Rand_M_bar_all = randperm(M);
    Rand_M_bar =unique(Rand_M_bar_all(1:M_bar)); % prendi i primi M_bar valori di M dopo aver mischiato i valori di M randomly
    disp([' Calculating for M_bar = ' num2str(M_bar)]);  

    % Keep only the coefficients of Ht that are related to M_bar
    Ht_bar = reshape(Ht(Rand_M_bar,:),M_bar*K_DL,1);

    %u_step=100;
    Htx=repmat(Ht(:,1),1,u_step); 
    % repmat: prende Ht array verticale e forma una nuova variabile Htx con shape 1, u_step.
    % in altre parole replica Ht lungo le colonne per u_step volte
    % Perchè prende solo la prima colonna di Ht, cioè il carrier 1 quando invece ce ne sono K_DL? (spiegato sotto nella formula del rate)

    % Per ogni regione verticale
    for pp = 1:1:numel(Ur_rows_grid)-1
        clear DeepMIMO_dataset 
        %disp(['Starting received user access ' num2str(pp)]);
        params.active_user_first=Ur_rows_grid(pp);
        params.active_user_last=Ur_rows_grid(pp+1)-1;
        filename_Hrpp=strcat(filename_Hr, '_pp', num2str(pp));
        params.saveDataset=0; % Non lo salvo perchè uguale a quello già calcolato in precedenza.
        % Senza ricalcolare il dataset, si potrebbe re-importarlo.
        % Forse occupava troppo spazio e in un pc portatile non ci stava, così hanno deciso di fare clear 
        % e ricalcolarla al bisogno.

        if load_mat_files == 1
            disp(['Loading ', filename_Hrpp, '...']);
            sfile_DeepMIMO=strcat(output_folder, 'DeepMIMO Dataset/DeepMIMO_dataset_', filename_Hrpp, '.mat');
            load(sfile_DeepMIMO);
            sfile_DeepMIMO=strcat(output_folder, 'DeepMIMO Dataset/params_', filename_Hrpp, '.mat');
            load(sfile_DeepMIMO);
            disp('Done');
        else
            params.saveDataset=1;
            tic
            params.filename = filename_Hrpp;
            disp(['Generating ', filename_Hrpp, '...']);
            [DeepMIMO_dataset,params]=DeepMIMO_generator(params);
            disp('Done');
            toc
        end

        Hrx=zeros(M,u_step);
        for u=1:u_step:params.num_user % Per ogni utente dentro una regione verticale a step di 100 (u_step)                   
            %disp([' params.num_user = ' num2str(params.num_user)]); % u_step*181 = 100*181 = 18100     
            
            %--- Input preparation
            for uu=1:1:u_step % A step di 1 dentro i 100 (praticamente potevano fare un unico loop invece di fare u e uu)
                Hr = single(conj(DeepMIMO_dataset{1}.user{u+uu-1}.channel)); % u+uu-1 ritorna l'indice di un utente dentro la grid complessiva
                %disp([' size(Hr) = ' num2str(size(Hr))]);  % size(Hr) = 1024, 64 = number of RIS elements, number of subcarriers

                % Keep only the coefficients of Hr that are related to M_bar --> Filtra per righe
                Hr_bar = reshape(Hr(Rand_M_bar,:),M_bar*K_DL,1); % reshape per ottenere un array
                %disp([' size(Hr_bar) = ' num2str(size(Hr_bar))]);  % size(Hr_bar) = 8x64= (512, 1)
                
                %--- Constructing the sampled channel
                n1=sqrt(noise_power_bar/2)*(randn(M_bar*K_DL,1)+1j*randn(M_bar*K_DL,1));
                n2=sqrt(noise_power_bar/2)*(randn(M_bar*K_DL,1)+1j*randn(M_bar*K_DL,1));
                H_bar = ((Ht_bar+n1).*(Hr_bar+n2)); % .* = element-wise multiplication = hadamard product
                %disp([' size(H_bar) = ' num2str(size(H_bar))]); % size(H_bar) = 512, 1 --> rimane sempre un array

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
                % numero link tra utenti e RIS = numero totale utenti = (1200 - 1000) x 181 = 36200
                DL_input(:,u+uu-1+((pp-1)*params.num_user)) = reshape([real(H_bar) imag(H_bar)].',[],1); % salva H_bar in
                %disp(['size(DL_input): ' num2str(size(DL_input))]); % size(DL_input) = 1024, 36200
                
                % location of the user pu = [px; py; pz] in the x-y-z space
                User_Location(:,u+uu-1+((pp-1)*params.num_user)) = single(DeepMIMO_dataset{1}.user{u+uu-1}.loc);
                
                Delta_H_bar = max(max(abs(H_bar))); 
                % Massimo di ogni colonna e poi massimo della riga risultate 
                % --> Si ottiene il valore massimo della matrice 2D, e questo massimo viene ricavato per tutti gli utenti e regioni
                % Delta_H_bar_max = denominatore formula 25 paper (pag 13)
                if Delta_H_bar >= Delta_H_bar_max
                    Delta_H_bar_max = single(Delta_H_bar);
                end
                Hrx(:,uu)=Hr(:,1); 
                % Perchè prende solo la prima colonna di Hr, cioè il carrier 1 quando invece ce ne sono K_DL? (spiegato sotto nella formula del rate)

            end

            %--- Actual achievable rate for performance evaluation
            H = Htx.*Hrx;
            %disp([' size(Ht(:,1)) = ' num2str(size(Ht(:,1)))]); % 1024, 1
            %disp([' size(Htx) = ' num2str(size(Htx))]); % 1024, 100
            %disp([' size(Hr(:,1)) = ' num2str(size(Ht(:,1)))]); % 1024, 1
            %disp([' size(Hrx) = ' num2str(size(Hrx))]); % 1024, 100
            %disp([' size(H) = ' num2str(size(H))]); % 1024, 100
            
            H_BF=H.'*BF_codebook;
            % Nel paper (pag 7): BF_codebook = psi piccolo = LIS interaction vector = reflection beamforming vector.
            %disp([' size(H_BF) = ' num2str(size(H_BF))]); % 100, 1024
            SNR_sqrt_var = abs(H_BF); 
            % commento francesco: indica il mismatch sulle fasi. se il codebook è perfetto, è un array di 1, altrimenti è minore di 1.
            % Sarà cmq minore di 1 per il path loss.
            %disp([' size(SNR_sqrt_var) = ' num2str(size(SNR_sqrt_var))]); % 100, 1024

            %keyboard;

            %--- Golden output preparation
            for uu=1:1:u_step
                if sum((Validation_Ind == u+uu-1+((pp-1)*params.num_user))) % per gli utenti che sono nel validation set, entra nell'if
                    count=count+1; % variabile inutile, potevano usare direttamente uu
                    %DL_output_un(count,:) = single( sum( log2( 1+( SNR*( (SNR_sqrt_var(uu,:)).^2 ) ) ), 1) );
                    DL_output_un(count,:) = single( log2( 1+( SNR*( SNR_sqrt_var(uu,:).^2 ) ) ) ); % sum not needed (Luca)
                    % DL_output_un è l'output del genie-aided

                    % Nel paper (pag 16) gli autori dichiarano di aver considerato un solo carrier per il calcolo del rate
                    % "to reduce the computational complexity of the performance evaluation". 
                    % Quindi rispetto alla formula 6 e 27 del paper (pag 7 e 16) manca la sommatoria sui carrier e la divisione per K.
                    % Questo è il motivo per cui quando generano Htx e Hrx ci inseriscono solo un carrier. Sono coerenti.
                    
                    %disp([' size(DL_output_un(count,:)):', num2str(size(DL_output_un(count,:)))]); % 1, 1024
                    % Poichè SNR_sqrt_var(uu,:) ritorna sempre un vettore (1, 1024) e il risultato finale DL_output_un(count,:) 
                    % è sempre un vettore (1, 1024), allora significa che la somma lunga la direzione 1 è inutile
                    %a = single( sum( log2( 1+( SNR*( (SNR_sqrt_var(uu,:)).^2 ) ) ), 1) );
                    %b = single( log2( 1+( SNR*( (SNR_sqrt_var(uu,:)).^2 ) ) ) );
                    %disp(['isequal(a, b): ', num2str(isequal(a, b))]);
                    
                    % Sono tutti diversi
                    %disp([' DL_output_un(count,1):', num2str(DL_output_un(count,1))]);
                    %disp([' DL_output_un(count,2):', num2str(DL_output_un(count,2))]);
                    %disp([' DL_output_un(count,100):', num2str(DL_output_un(count,100))]);
                    %disp([' DL_output_un(count,1024):', num2str(DL_output_un(count,1024))]);
                    %keyboard;
                end
            end

            %--- Target output preparation
            %--- Label for the sampled channel
            R = single(log2(1+(SNR_sqrt_var/Delta_H_max).^2));
            %disp([' size(R):', num2str(size(R))]); % 100, 1024
            % Ogni riga corrisponde a un vector of rates. Il numero di rates è pari al codebook size = 1024. Ogni rate è il risultato dell'aver utilizzato un certo codebook.
            
            % Questa formula è simile al rate ma manca la moltiplicazione per il SNR e viene divisa per Delta_H_max.
            % Che cosa rappresenta questa formula?
            % Può essere un forma del rate normalizzata rispetto al SNR perchè avrebbero dovuto moltiplicarlo,
            % ma poi anche dividerlo perciò si elide?
            % E' come se fosse un rate approssimato usato per il DL, mentre quello "vero" è quello in DL_output_un

            % Perchè normalizzano rispetto a Delta_H_max?

            %--- DL output normalization
            Delta_Out_max = max(R,[],2); % 100, 1
            % returns the maximum element along dimension 2, i.e., returns a column vector containing the maximum value of each row.
            
            %disp([' size(Delta_Out_max) = ' num2str(size(Delta_Out_max))]); % 100, 1

            if ~sum(Delta_Out_max == 0) % Se non ci sono elementi nulli
                % Nel paper (pag 14) dice che every vector of rates r(s) is normalized using its maximum rate value (per-sample normalization).
                % Dicono che serve per avere un modello che non sia biased towards some strong responses, i.e.,
                % it gives the receivers equal importance regardless of how close or far they are from the LIS.
                % Quindi ogni vettore di rates avrà come valore massimo il numero 1.
                Rn=diag(1./Delta_Out_max)*R;
                % diag(v) restituisce una matrice diagonale quadrata con gli elementi del vettore v sulla diagonale principale.
                % Each row of R is scaled by the corresponding diagonal element

            end
            DL_output(u+((pp-1)*params.num_user):u+((pp-1)*params.num_user)+u_step-1,:) = Rn; % scrive una matrice 100, 1024 alla volta

            %disp([' size(R) = ' num2str(size(R))]); % 100, 1024
            %disp([' size(Rn) = ' num2str(size(Rn))]); % 100, 1024
            %disp([' size(DL_output) = ' num2str(size(DL_output))]); % 36200, 1024

            %keyboard;
            
        end

        %keyboard; 
    end
    clear u Delta_H_bar R Rn

    %disp([' count:', num2str(count)]); % 6200

    %-- Sorting back the DL_output_un
    DL_output_un = DL_output_un(VI_rev_sortind,:);

    %--- DL input normalization (come da formula 25 paper pag 13)
    %DL_input= 1*(DL_input/Delta_H_bar_max); %%%%% Normalized from -1->1 %%%%% (commento originale)
    % Secondo me il commento è sbagliato perchè non si sa se gli input saranno tra -1 e 1, ma sappiamo solo che il massimo sarà 1.
    DL_input= DL_input/Delta_H_bar_max; % Normalized to 1 (LUCA)
    % Non so perchè hanno scritto questo commento alla riga sopra, però dividere per Delta_H_bar_max
    % significa che il numero massimo dentro a DL_input diventa 1, ma i numeri non cambiano da negativi a positivi,
    % a meno che Delta_H_bar_max non sia negativo, ma ho verificato che non è così (8.2934e-12).
    % non possiamo garantire che i valori saranno compresi tra -1 e 1 a meno che 
    % DL_input non contenga valori negativi e positivi simmetrici rispetto a zero. E' possibile?
    disp([' Delta_H_bar_max:', num2str(Delta_H_bar_max)]);
    %disp([' size(Delta_H_bar_max):', num2str(size(Delta_H_bar_max))]); % 1, 1

    %keyboard; % Pause

    %disp([' size(DL_input) = ' num2str(size(DL_input))]);
    %disp([' size(DL_output) = ' num2str(size(DL_output))]);

    if save_mat_files == 1
        save(filename_DL_input,'DL_input','-v7.3');
        save(filename_DL_output,'DL_output','-v7.3');
        save(filename_DL_output_un,'DL_output_un','-v7.3');   
        save(filename_User_Location,'User_Location','-v7.3');
        save(filename_params,'params','-v7.3');
    end
end

%keyboard; % Pause

%% DL Beamforming

disp('---> DL Beamforming');



if load_Rates == 1
    disp('Loading Rate_DL, Rate_OPT...');
    load(filename_Rate_DL);
    load(filename_Rate_OPT);
    disp('Done');
else

    miniBatchSize  = 500; % Size of the minibatch for the Deep Learning

    % Preallocation of output variables
    Rate_DL = zeros(1,length(Training_Size)); 
    Rate_OPT = Rate_DL;
    LastValidationRMSE = Rate_DL;
    Rate_DL_fake = Rate_DL; % Luca
    validation_accuracy = Rate_DL; % Luca

    % ------------------ Training and Testing Datasets -----------------%
    DL_output_reshaped = reshape(DL_output.',1,1,size(DL_output,2),size(DL_output,1)); % 1, 1, 1024, 36200
    DL_output_reshaped_un = reshape(DL_output_un.',1,1,size(DL_output_un,2),size(DL_output_un,1));
    DL_input_reshaped = reshape(DL_input,size(DL_input,1),1,1,size(DL_input,2));

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

        %disp([' size(XTrain) = ' num2str(size(XTrain))]); % 1024, 1, 1, Training_Ind(dd)
        %disp([' size(YTrain) = ' num2str(size(YTrain))]); % 1, 1, 1024, Training_Ind(dd)
        %disp([' size(XValidation) = ' num2str(size(XValidation))]); % 1024, 1, 1, 6200
        %disp([' size(YValidation) = ' num2str(size(YValidation))]); % 1, 1, 1024, 6200

        %keyboard;
        
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
            % Il layer di regressione utilizza i minimi quadrati (mean squared error, MSE)
            % come funzione di perdita per impostazione predefinita.

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

        % ------------- DL Model Training and Prediction -----------------%
        tic
        disp('Start DL training...')
        [trainedNet,traininfo]  = trainNetwork(XTrain,YTrain,layers,options);    
        disp('Done')
        toc;
        elapsedTime = toc; % Misura il tempo trascorso in secondi
        elapsedTimeMinutes = elapsedTime / 60; % Converti il tempo trascorso in minuti
        disp(['Elapsed time is ', num2str(elapsedTimeMinutes), 'minutes']);

        sfile_DeepMIMO=strcat(filename_trainedNet, '_Training_Size_', num2str(Training_Size(dd)), '.mat');
        save(sfile_DeepMIMO,'trainedNet','-v7.3');
        
        %keyboard;

        tic
        disp('Start DL prediction for Figure 12...')
        YPredicted = predict(trainedNet,XValidation); % Inferenza sul set di validazione usato come test: errore!
        disp('Done')
        toc

        disp(['size(YPredicted) = ' num2str(size(YPredicted))]); % 6200, 1024
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
            % YValidation_un = DL_output_reshaped_un = DL_output_un
            %MaxR_DL(b) = max(squeeze(YValidation_un(1,1,Indmax_DL(b,:),b))); %True achievable rates
            MaxR_DL(b) = squeeze(YValidation_un(1,1,Indmax_DL(b),b)); %True achievable rates (Luca)
            MaxR_OPT(b) = squeeze(YValidation_un(1,1,Indmax_OPT(b),b));

            % Count the number of correct predictions (Luca)
            if MaxR_DL(b) == MaxR_OPT(b)
                validation_accuracy(dd) = validation_accuracy(dd) + 1;
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
        disp(' ');

        %keyboard;

        clear trainedNet traininfo YPredicted
        clear layers options 
        
    end

    if save_mat_files == 1
        save(filename_DL_input_reshaped, 'DL_input_reshaped', '-v7.3');
        save(filename_DL_output_reshaped, 'DL_output_reshaped', '-v7.3');
        save(filename_Rate_DL,'Rate_DL','-v7.3');
        save(filename_Rate_OPT,'Rate_OPT','-v7.3');
    end
end


%% Fig 7 (Luca)

disp('---> Plot Fig 7');

if plot_fig7 == 1
    
    disp('Loading DL_input_reshaped, DL_output_reshaped, trainedNet, loc, ...');  

    % Concatena XTrain + XValidation, cioè utilizza DL_input_reshaped
    load(filename_DL_input_reshaped);
    % Concatena YTrain + YValidation, cioè utilizza DL_output_reshaped
    load(filename_DL_output_reshaped);

    % Carica modello pre-allenato caso Training_Size = 30000
    dd = 1;
    sfile_DeepMIMO=strcat(filename_trainedNet, '_Training_Size_', num2str(Training_Size(dd)), '.mat');
    trainedNet = load(sfile_DeepMIMO).trainedNet;

    if save_mat_files == 1
        % Esegui predizione con DL_input_reshaped
        tic
        disp('Start DL prediction for Figure 7...')
        YPredictedFig7 = predict(trainedNet,DL_input_reshaped); % Inferenza sul set di validazione usato come test: errore!
        disp('Done')
        toc
        save(filename_YPredictedFig7,'YPredictedFig7','-v7.3'); 
    else
        load(filename_YPredictedFig7);
    end

    % Recupera gli indici dei codebook
    [~,Indmax_OPT] = max(DL_output_reshaped,[],3);
    %disp(['size(Indmax_OPT):', num2str(size(Indmax_OPT))]); % 1, 1, 1, 36200
    Indmax_OPT = squeeze(Indmax_OPT);
    %disp(['size(Indmax_OPT):', num2str(size(Indmax_OPT))]); % 36200, 1
    Indmax_OPT = Indmax_OPT.'; % 1, 36200
    
    [~,Indmax_DL] = maxk(YPredictedFig7,kbeams,2);
    %disp(['size(Indmax_DL):', num2str(size(Indmax_DL))]); % 36200, 1

    % Recupera locations loc
    load(filename_User_Location);

    User_Location_norm = single(zeros(3,No_user_pairs));
    space_between_users = 0.2; % meters
    for i=1:1:size(User_Location_norm,1)-1 % Lavora solo su x e y perchè z è costante
        User_Location_norm(i, :) = round(( User_Location(i,:) - min(User_Location(i,:)) ) / space_between_users + 1);
        %disp(min(User_Location(i,:)));
        %disp(min(User_Location_norm(i,:)));
        %disp(max(User_Location(i,:)));
        %disp(max(User_Location_norm(i,:)));
    end
    %keyboard;
        
    % Grafico
    % Il sistema di riferimento x,y è quello del grafico, NON quello dello scenario.
    reversed_y = 1:200; %User_Location_norm(2, 1:181:end); % prendere un valore ogni 181, risultando in un vettore da 1 a 200
    reversed_x = 1:181; %User_Location_norm(1, 1:max(User_Location_norm(1,:))); % prendere i primi 181 valori, risultando in un vettore da 1 a 181
    %disp(size(reversed_y)); % 200
    %disp(size(reversed_x)); % 181

    % Matrice dei valori
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

    %values = reshape(Indmax_OPT, numel(reversed_y), []).'; % 181x200 SBAGLIATO
    % Il secondo argomento specifica il numero di righe desiderato (che diventeranno le colonne con la trasposizione finale).
    % Il terzo argomento [] lascia che MATLAB calcoli automaticamente il numero di colonne necessario in base alla lunghezza del vettore Indmax_OPT.
    % Trasposta la matrice risultante per ottenere 181x200.

    values1 = reshape(Indmax_OPT, numel(reversed_x), []); % 181x200 QUASI GIUSTO: perchè?
    % MATLAB divide il vettore Indmax_OPT in blocchi di lunghezza pari a [] e 
    % li organizza in numel(reversed_x) righe.
    % I primi [] elementi di Indmax_OPT diventano la prima riga della matrice.
    % I successivi [] elementi diventano la seconda riga, e così via.
    % Il numero di colonne [] viene calcolato come length(Indmax_OPT) / numel(reversed_x).
    values_DL = reshape(Indmax_DL, numel(reversed_x), []); % 181x200 QUASI GIUSTO: perchè?
    
    % EQUIVALENTE A RESHAPE LINEA 784
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

    values_OPT = values1;

    % Creazione del grafico
    f7 = figure('Name', 'Figure7', 'units','pixels', 'Position', [100, 100, 2400, 350]); % typ 800x400

    % Calcolo minimi e massimi colorbar comuni ai due grafici
    min_colorbar = min([min(values_OPT(:)), min(values_DL(:))]);
    max_colorbar = max([max(values_OPT(:)), max(values_DL(:))]);

    % Subplot 1
    subplot(1, 2, 1); % 1 riga, 2 colonne, primo subplot
    imagesc([1 numel(reversed_y)], [numel(reversed_x) 1], values_OPT); 
    % Nella prima quadra ci sono le coordinate x, nella seconda le coordinate y. Bisogna accoppiarle element-wise, quindi:
    % (1,reversed_x)=(1,181) corresponds to values(1,1)
    % (reversed_y,1)=(200,1) corresponds to values(200,181)
    % Plotta la matrice values mappando ogni elemento a un pixel del grafico.
    colormap(parula); % Imposta la colormap arcobaleno
    colorbar; % Aggiunge una barra dei colori
    caxis([min_colorbar, max_colorbar]); % Imposta i limiti della scala dei colori
    xlabel('Horizontal direction (reversed y-axis)');
    ylabel('Vertical direction (reversed x-axis)');
    title('(a) Original Codebook Beams');

    % Subplot 2
    subplot(1, 2, 2); % 1 riga, 2 colonne, secondo subplot
    imagesc([1 numel(reversed_y)], [numel(reversed_x) 1], values_DL); 
    % Nella prima quadra ci sono le coordinate x, nella seconda le coordinate y. Bisogna accoppiarle element-wise, quindi:
    % (1,reversed_x)=(1,181) corresponds to values(1,1)
    % (reversed_y,1)=(200,1) corresponds to values(200,181)
    % Plotta la matrice values mappando ogni elemento a un pixel del grafico.
    colormap(parula); % Imposta la colormap arcobaleno
    colorbar; % Aggiunge una barra dei colori
    caxis([min_colorbar, max_colorbar]); % Imposta i limiti della scala dei colori
    xlabel('Horizontal direction (reversed y-axis)');
    ylabel('Vertical direction (reversed x-axis)');
    title('(b) Predicted Codebook Beams');

    sfile_DeepMIMO=strcat(output_folder, 'Fig7', '.png');
    saveas(f7, sfile_DeepMIMO);
    close(f7);

    keyboard;

end

%% End of script

disp('--> End of script. If you continue, you will lose access to all variables');

%keyboard;

end
