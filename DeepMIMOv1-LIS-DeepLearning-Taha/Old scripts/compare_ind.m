% Leggi i due file txt
file1 = 'C:\Users\Work\Desktop\deepMIMO\RIS\DeepMIMOv1-LIS-DeepLearning-Taha\Output_Python\Figures\Indmax_OPT_mat.txt';
file2 = 'C:\Users\Work\Desktop\deepMIMO\RIS\DeepMIMOv1-LIS-DeepLearning-Taha\Output_Python\Figures\Indmax_OPT_py.txt';

data1 = load(file1); % Carica i dati dal primo file
data2 = load(file2); % Carica i dati dal secondo file

% Calcola la differenza elemento per elemento
difference = data1 - data2;

for i=1:100
    disp([data1(i), data2(i), difference(i)])
end

% Salva la differenza in un nuovo file
output_file = 'C:\Users\Work\Desktop\deepMIMO\RIS\DeepMIMOv1-LIS-DeepLearning-Taha\Output_Python\Figures\difference_Ind.txt';
writematrix(difference, output_file);

% Calcola il minimo e il massimo della differenza
min_diff = min(difference);
max_diff = max(difference);

% Stampa i risultati
disp(['Minimo della differenza: ', num2str(min_diff)]);
disp(['Massimo della differenza: ', num2str(max_diff)]);