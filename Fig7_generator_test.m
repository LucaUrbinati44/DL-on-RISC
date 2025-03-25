% Dati di esempio
x = 1:200; % Coordinate orizzontali (direzione x)
y = 1:180; % Coordinate verticali (direzione y)

% Matrice dei valori (esempio con una funzione radiale)
[X, Y] = meshgrid(x, y); % Crea una griglia di coordinate
values = sqrt((X - 100).^2 + (Y - 90).^2); % Distanza radiale dal centro (100, 90)
disp(size(values));
%values = rand(1, 400);

% Creazione del grafico
f7 = figure('Name', 'Figure7', 'units','pixels');
imagesc(x, y, values); % Display image with scaled colors
set(gca, 'YDir', 'reverse'); % Inverte l'asse y per corrispondere al grafico
colormap(jet); % Imposta la colormap arcobaleno
colorbar; % Aggiunge una barra dei colori
caxis([min(values(:)), max(values(:))]); % Imposta i limiti della scala dei colori

% Etichette e titolo
xlabel('Horizontal direction (reversed y-axis)');
ylabel('Vertical direction (reversed x-axis)');
title('(a) Original Codebook Beams');

sfile_DeepMIMO=strcat('./Fig7test', '.png');
saveas(f7, sfile_DeepMIMO);
close(f7);

keyboard;