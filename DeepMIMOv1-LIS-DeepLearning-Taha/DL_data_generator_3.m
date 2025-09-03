function [RandP_all,Validation_Ind]=DL_data_generator_3(Mx,My,Mz,M_bar,D_Lambda,BW,K,K_DL,Pt,Ur_rows,Ut_element,Ur_rows_grid,Validation_Size)
    
%% Deep Learning Dataset Generation = Genie-Aided Reflection Beamforming

disp('---> Deep Learning Dataset Generation');

global load_H_files load_Delta_H_max load_DL_dataset load_Rates save_mat_files;
global seed DeepMIMO_dataset_folder end_folder end_folder_M_bar DL_dataset_folder network_folder figure_folder;

filename_Ht=strcat(DeepMIMO_dataset_folder, 'Ht', '_seed', '_grid', num2str(seed), num2str(Ur_rows(2)), '_M', num2str(My), num2str(Mz), '.mat');
filename_params_Ht=strcat(DeepMIMO_dataset_folder, 'params_Ht', end_folder, '.mat');
filename_Hr=strcat(DeepMIMO_dataset_folder, 'Hr', end_folder);
filename_params_Hr=strcat(DeepMIMO_dataset_folder, 'params_Hr', end_folder);
filename_Delta_H_max=strcat(DeepMIMO_dataset_folder, 'Delta_H_max', end_folder, '.mat');
filename_User_Location=strcat(DeepMIMO_dataset_folder, 'User_Location', end_folder_M_bar, '.mat');
%filename_params=strcat(DeepMIMO_dataset_folder, 'params', end_folder_M_bar, '.mat');

filename_RandP_all=strcat(DL_dataset_folder, 'RandP_all', end_folder_M_bar, '.mat');
filename_DL_input=strcat(DL_dataset_folder, 'DL_input', end_folder_M_bar, '.mat');
filename_DL_output=strcat(DL_dataset_folder, 'DL_output', end_folder_M_bar, '.mat');
filename_DL_output_un=strcat(DL_dataset_folder, 'DL_output_un', end_folder_M_bar, '.mat');

filename_DL_input_reshaped=strcat(DL_dataset_folder, 'DL_input_reshaped', end_folder_M_bar, '.mat');
filename_DL_output_reshaped=strcat(DL_dataset_folder, 'DL_output_reshaped', end_folder_M_bar, '.mat');
filename_DL_output_un_reshaped=strcat(DL_dataset_folder, 'DL_output_un_reshaped', end_folder_M_bar, '.mat');
filename_DL_output_un_complete_reshaped=strcat(DL_dataset_folder, 'DL_output_un_complete_reshaped', end_folder_M_bar, '.mat');

% Note: The axes of the antennas match the axes of the ray-tracing scenario
M = Mx.*My.*Mz; % Total number of LIS reflecting elements 

No_user_pairs = (Ur_rows(2)-Ur_rows(1))*181; % Number of users (= Number of TX-RX pairs) = Total numer of users in Grid 1
%disp([' No_user_pairs = ' num2str(No_user_pairs)]);
% In the 'O1' scenario where every row consists of 181 points.
% Since the number of BS antennas is one, the number of pairs is equal to the number of users.
RandP_all = randperm(No_user_pairs).';
%save(filename_RandP_all,'RandP_all','-v7.3');
%return

% randperm(n) restituisce un vettore contenente una permutazione casuale di numeri interi da 1 a n, senza elementi ripetuti.
% Serve per ottenere gli indici mischiati (shuffled)

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
    DL_output_un_complete =  single(zeros(No_user_pairs,codebook_size));
    Delta_H_bar_max = single(0);
    count=0;

    % The active channel sensors are randomly selected from the M UPA antennas
    Rand_M_bar_all = randperm(M);
    Rand_M_bar =unique(Rand_M_bar_all(1:M_bar)); % prendi i primi M_bar valori di M dopo aver mischiato i valori di M randomly
    disp([' Calculating for M_bar = ' num2str(M_bar)]);  

    if load_Delta_H_max == 1
        disp('Loading Delta_H_max...');
        load(filename_Delta_H_max);
        disp('Done');
    end

    if load_H_files == 1
        disp(['Loading ', filename_Ht, '...']);
        load(filename_Ht);
        load(filename_params_Ht);
        disp('Done');

        Ht = single(DeepMIMO_dataset{1}.user{Ut_element}.channel);
        %disp([' size(DeepMIMO_dataset) = ' num2str(size(DeepMIMO_dataset))]); % returns 1
        %disp([' size(Ht) = ' num2str(size(Ht))]); % size(Ht) = 1024, 64
        %keyboard; 
        clear DeepMIMO_dataset
    end    
    
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
        filename_Hrpp=strcat(filename_Hr, '_pp', num2str(pp), '.mat');
        filename_params_Hrpp=strcat(filename_params_Hr, '_pp', num2str(pp), '.mat');
        params.saveDataset=0; % Non lo salvo perchè uguale a quello già calcolato in precedenza.
        % Senza ricalcolare il dataset, si potrebbe re-importarlo.
        % Forse occupava troppo spazio e in un pc portatile non ci stava, così hanno deciso di fare clear 
        % e ricalcolarla al bisogno.

        if load_H_files == 1
            disp(['Loading ', filename_Hrpp, '...']);
            load(filename_Hrpp);
            load(filename_params_Hrpp);
            disp('Done');
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

                DL_output_un_complete(u+uu-1+((pp-1)*params.num_user), :) = single( log2( 1+( SNR*( SNR_sqrt_var(uu,:).^2 ) ) ) ); % Luca
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

    DL_input_reshaped = reshape(DL_input,size(DL_input,1),1,1,size(DL_input,2)); % 1024, 1, 1, 36200
    DL_output_reshaped = reshape(DL_output.',1,1,size(DL_output,2),size(DL_output,1)); % 1, 1, 1024, 36200
    DL_output_un_reshaped = reshape(DL_output_un.',1,1,size(DL_output_un,2),size(DL_output_un,1));
    DL_output_un_complete_reshaped = reshape(DL_output_un_complete.',1,1,size(DL_output_un_complete,2),size(DL_output_un_complete,1));

    %keyboard;

    %save(filename_DL_output_un_complete_reshaped, 'DL_output_un_complete_reshaped', '-v7.3');

    if save_mat_files == 1

        save(filename_RandP_all,'RandP_all','-v7.3');

        save(filename_DL_input,'DL_input','-v7.3');
        save(filename_DL_output,'DL_output','-v7.3');
        save(filename_DL_output_un,'DL_output_un','-v7.3');   
        
        save(filename_DL_input_reshaped, 'DL_input_reshaped', '-v7.3');
        save(filename_DL_output_reshaped, 'DL_output_reshaped', '-v7.3');
        save(filename_DL_output_un_reshaped, 'DL_output_un_reshaped', '-v7.3');
        save(filename_DL_output_un_complete_reshaped, 'DL_output_un_complete_reshaped', '-v7.3');
        
        save(filename_User_Location,'User_Location','-v7.3');
        %save(filename_params,'params','-v7.3');
    end
end

%keyboard; % Pause