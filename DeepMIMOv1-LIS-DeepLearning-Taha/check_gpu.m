clc
close all
clear all

% Definisci i valori di n in progressione logaritmica
n_values = [100, 1000, 5000, 10000, 20000]; % Variazione logaritmica di n
%n_values = [1000, 2000];
cpu_times = zeros(size(n_values)); % Array per memorizzare i tempi su CPU
gpu_times = zeros(size(n_values)); % Array per memorizzare i tempi su GPU

compute = 1;

if compute == 1
    % Ciclo per ogni dimensione n
    for i = 1:length(n_values)
        n = n_values(i);  % Ottieni la dimensione corrente
        
        % Crea le matrici casuali A e B
        A = rand(n, 'single');
        B = rand(n, 'single');
        
        % Esecuzione su CPU
        disp(['Esecuzione su CPU per n = ', num2str(n)]);
        tic; % Iniziamo il cronometro
        C_cpu = A * B; % Operazione di moltiplicazione matrice su CPU
        cpu_times(i) = toc; % Memorizza il tempo su CPU
        
        % Esecuzione su GPU
        disp(['Esecuzione su GPU per n = ', num2str(n)]);
        A_gpu = gpuArray(A); % Trasferisci A sulla GPU
        B_gpu = gpuArray(B); % Trasferisci B sulla GPU
        tic; % Iniziamo il cronometro per la GPU
        C_gpu = A_gpu * B_gpu; % Operazione di moltiplicazione matrice su GPU
        % Trasferiamo il risultato dalla GPU alla CPU (per visualizzarlo se necessario)
        C_gpu_cpu = gather(C_gpu);
        gpu_times(i) = toc; % Memorizza il tempo su GPU

        % Verifica se i risultati sono uguali (approssimativamente)
        if norm(C_cpu - C_gpu_cpu) < 1e-4
            disp('I risultati su CPU e GPU sono uguali.');
        else
            disp('I risultati su CPU e GPU sono diversi.');
        end
    end

    save('./cpu_times.mat','cpu_times','-v7.3');
    save('./gpu_times.mat','gpu_times','-v7.3');

end

load('./cpu_times.mat');
load('./gpu_times.mat');

% Grafico dei risultati
fig = figure();
semilogx(n_values, cpu_times, '-o', 'LineWidth', 2, 'MarkerSize', 8); % Tempo CPU
hold on;
semilogx(n_values, gpu_times, '-s', 'LineWidth', 2, 'MarkerSize', 8); % Tempo GPU
hold off;

% Etichette e titolo
xlabel('Dimensione matrice (n)', 'FontSize', 12);
ylabel('Tempo di esecuzione (secondi)', 'FontSize', 12);
title('Prodotto tra matrici: CPU vs GPU', 'FontSize', 14);
legend({'CPU', 'GPU'}, 'Location', 'NorthWest');
grid on;

path=strcat('./TestGPU_matlab.png');
saveas(fig, path);
%close(fig);

% Calcola lo speedup
speedup = zeros(size(n_values));
for i = 1:length(n_values)
    speedup(i) = cpu_times(i) / gpu_times(i);
    disp(['Speedup GPU rispetto a CPU: ', num2str(speedup(i))]);
end

% Grafico dello speedup
fig2 = figure();
semilogx(n_values, speedup, '-d', 'LineWidth', 2, 'MarkerSize', 8, 'Color', 'r'); % Speedup
xlabel('Dimensione matrice (n)', 'FontSize', 12);
ylabel('Speedup (CPU / GPU)', 'FontSize', 12);
title('Speedup della GPU rispetto alla CPU', 'FontSize', 14);
grid on;

% Salvataggio del grafico in un file
path2 = './SpeedupGPU_matlab.png';  % Percorso per salvare l'immagine
saveas(fig2, path2);  % Salva la figura in un file
%close(fig2);  % Chiudi la figura
