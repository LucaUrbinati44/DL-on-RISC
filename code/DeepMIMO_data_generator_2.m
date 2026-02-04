function [Ur_rows_grid]=DeepMIMO_data_generator_2(Mx,My,Mz,D_Lambda,BW,K,K_DL,L,Ut_row,Ut_element,Ur_rows,params)

%% DeepMIMO Dataset Generation

disp('---> DeepMIMO Dataset Generation');

global load_H_files load_Delta_H_max load_DL_dataset load_Rates save_mat_files;
global seed DeepMIMO_dataset_folder end_folder DL_dataset_folder network_folder figure_folder;

filename_Ht=strcat(DeepMIMO_dataset_folder, 'Ht', end_folder, '.mat');
filename_params_Ht=strcat(DeepMIMO_dataset_folder, 'params_Ht', end_folder, '.mat');
filename_Hr=strcat(DeepMIMO_dataset_folder, 'Hr', end_folder);
filename_params_Hr=strcat(DeepMIMO_dataset_folder, 'params_Hr', end_folder);
filename_Delta_H_max=strcat(DeepMIMO_dataset_folder, 'Delta_H_max', end_folder, '.mat');

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
%disp([' Calculating for K_DL = ' num2str(K_DL)]);          
%disp([' Calculating for L = ' num2str(params.num_paths)]);

% ------------------ DeepMIMO "Ut" Dataset Generation -----------------%
params.active_user_first=Ut_row; 
params.active_user_last=Ut_row;

Ur_rows_step = 100; % access the dataset 100 rows at a time
% indica la posizione del RX lungo asse y, cioè la sua posizione nella riga, quindi la colonna della griglia degli utenti
Ur_rows_grid=Ur_rows(1):Ur_rows_step:Ur_rows(2); % [1000 1100 1200] --> [1000 1100 1200 1300] 

% ------------------ DeepMIMO "Delta_H_max" Generation -----------------%

if load_Delta_H_max == 1
    disp('Loading Delta_H_max...');
    load(filename_Delta_H_max);
    disp('Done');
else

    if load_H_files == 1
        disp(['Loading ', filename_Ht, '...']);
        load(filename_Ht);
        load(filename_params_Ht);
        disp('Done');
    else
        tic
        disp(['Generating ', filename_Ht, '...']);
        DeepMIMO_dataset=DeepMIMO_generator(params);
        disp('Done');
        toc

        if save_mat_files == 1
            save(filename_Ht,'DeepMIMO_dataset','-v7.3'); % save(filename, variables, options), 
            save(filename_params_Ht,'params','-v7.3');
            % -v7.3 è utilizzato per salvare file MAT che possono contenere variabili di grandi dimensioni e supporta la compressione.
        end
    end
    
    % .channel restituisce la matrice di canale tra l'active base station {1}, che in realtà è la RIS
    % e l'utente {Ut_element} che in realtà è la BS trasmettitore
    Ht = single(DeepMIMO_dataset{1}.user{Ut_element}.channel);
    %disp([' size(DeepMIMO_dataset) = ' num2str(size(DeepMIMO_dataset))]); % returns 1
    %disp([' size(Ht) = ' num2str(size(Ht))]); % size(Ht) = 1024, 64
    %keyboard; 
    clear DeepMIMO_dataset

    % ------------------ DeepMIMO "Ur" Dataset Generation -----------------%            
    Delta_H_max = single(0); % initilizzato a 0

    % Divide la griglia in tre regioni verticali e itera su ognuna di esse
    for pp = 1:1:numel(Ur_rows_grid)-1 % 3-1=2, loop for Normalizing H
        clear DeepMIMO_dataset
        params.active_user_first=Ur_rows_grid(pp);
        params.active_user_last=Ur_rows_grid(pp+1)-1; 
        filename_Hrpp=strcat(filename_Hr, '_pp', num2str(pp), '.mat');
        filename_params_Hrpp=strcat(filename_params_Hr, '_pp', num2str(pp), '.mat');

        if load_H_files == 1
            disp(['Loading ', filename_Hrpp, '...']);
            load(filename_Hrpp);
            load(filename_params_Hrpp);
            disp('Done');
        else
            tic
            disp(['Generating ', filename_Hrpp, '...']);
            [DeepMIMO_dataset,params]=DeepMIMO_generator(params);
            disp('Done');
            toc

            if save_mat_files == 1
                save(filename_Hrpp,'DeepMIMO_dataset','-v7.3');
                save(filename_params_Hrpp,'params','-v7.3');
            end
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

        %disp([' size(Hr) = ' num2str(size(Hr))]);
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
