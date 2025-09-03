file_type = 'full';

% Nome del file di input
input_file = ['C:\Users\Work\Desktop\deepMIMO\RIS\DeepMIMOv1-LIS-DeepLearning-Taha\Output_Python\Figures\YValidation_un_complete_',file_type,'.txt'];

% Leggi la matrice dal file CSV
matrix = readmatrix(input_file);

% Numero di righe nella matrice
[num_rows, num_cols] = size(matrix);

% Inizializza una matrice per salvare i risultati
% Ogni riga conterrà: [valore precedente, valore massimo, valore successivo]
results = nan(num_rows, 3);

% Inizializza un vettore per salvare le differenze
differences = nan(num_rows, 1);

% Per ogni riga, trova il massimo e i valori vicini
for i = 1:num_rows
    % Trova l'indice del valore massimo nella riga
    [max_val, max_idx] = max(matrix(i, :));
    
    % Trova il valore precedente (se esiste)
    if max_idx > 1
        prev_val = matrix(i, max_idx - 1);
    else
        prev_val = NaN; % Nessun valore precedente
    end
    
    % Trova il valore successivo (se esiste)
    if max_idx < num_cols
        next_val = matrix(i, max_idx + 1);
    else
        next_val = NaN; % Nessun valore successivo
    end
    
    % Calcola il minimo tra i due valori vicini
    min_neighbor = min([prev_val, next_val], [], 'omitnan');
    
    % Calcola la differenza tra il massimo e il minimo dei vicini
    diff = max_val - min_neighbor;
    
    % Salva i risultati
    results(i, :) = [prev_val, max_val, next_val];
    differences(i) = diff;
end

% Nome dei file di output
output_file_results = ['C:\Users\Work\Desktop\deepMIMO\RIS\DeepMIMOv1-LIS-DeepLearning-Taha\Output_Python\Figures\max_and_neighbors_',file_type,'.txt'];
output_file_differences = ['C:\Users\Work\Desktop\deepMIMO\RIS\DeepMIMOv1-LIS-DeepLearning-Taha\Output_Python\Figures\differences_neighbors_yvalidation_',file_type,'.txt'];

% Salva i risultati in un file CSV
writematrix(results, output_file_results);

% Salva le differenze in un file separato
writematrix(differences, output_file_differences);

% Calcola il massimo e il minimo delle differenze
min_diff = min(differences);
max_diff = max(differences);

% Stampa i risultati
disp(['Minimo della differenza: ', num2str(min_diff)]);
disp(['Massimo della differenza: ', num2str(max_diff)]);
disp(['Risultati salvati in ', output_file_results]);
disp(['Differenze salvate in ', output_file_differences]);

%% Genera un grafico per il numero di righe che superano la soglia
%thresholds = linspace(min_diff, max_diff, 100); % Genera 100 valori tra min_diff e max_diff
%rows_above_threshold = arrayfun(@(th) sum(differences <= th), thresholds)/numel(differences)*100;
%
%% Crea il grafico
%figure;
%plot(thresholds, rows_above_threshold, 'b-', 'LineWidth', 2);
%xlabel('Soglia');
%ylabel('Numero di rate rispetto al totale [%]');
%title({['Numero di rate vicini al massimo di una posizione'] ['la cui differenza con il massimo è superiore o uguale alla soglia']});
%grid on;
%
%% Salva il grafico in formato PNG
%output_plot_file = ['C:\Users\Work\Desktop\deepMIMO\RIS\DeepMIMOv1-LIS-DeepLearning-Taha\Output_Python\Figures\differences_threshold_plot_',file_type,'.png'];
%saveas(gcf, output_plot_file);
%
%disp(['Grafico salvato in ', output_plot_file]);


% Genera i bin per la distribuzione
num_bins = 5; % Numero di intervalli (modificabile)
edges = linspace(min_diff, max_diff, num_bins + 1); % Limiti dei bin
bin_counts = histcounts(differences, edges); % Conta i valori in ciascun bin

% Calcola la percentuale per ciascun bin
bin_percentages = (bin_counts / sum(bin_counts)) * 100;

% Crea il grafico a barre
figure;
bar(edges(1:end-1), bin_percentages, 'histc'); % Usa 'histc' per allineare le barre ai bordi
xlabel('Differenza tra massimo e minimo dei vicini one-step-away dal massimo');
ylabel('Percentuale (%)');
%title('Distribuzione delle differenze');
%title('Distribuzione del numero di rate vicini al massimo di una posizione');
title({['Distribuzione della differenza tra il massimo'] ['e il minimo dei vicini che sono one-step-away dal massimo']});
grid on;

% Salva il grafico in formato PNG
output_bar_plot_file = ['C:\Users\Work\Desktop\deepMIMO\RIS\DeepMIMOv1-LIS-DeepLearning-Taha\Output_Python\Figures\differences_distribution_plot_',file_type,'.png'];

saveas(gcf, output_bar_plot_file);

disp(['Grafico a barre salvato in ', output_bar_plot_file]);