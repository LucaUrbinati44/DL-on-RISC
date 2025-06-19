function []=FigD_plot_all_tflite(My_ar,Mz_ar,M_bar,Ur_rows,kbeams,Training_Size,Validation_Ind,Test_Ind,epochs,plot_mode,Training_Size_number,M_ar_master, ...
                                MaxR_OPTt,MaxR_DLt_mat, ...
                                MaxR_OPTt_py_test, ...
                                MaxR_DLt_py_test_20,  MaxR_DLt_py_test_tflite_20, ...
                                MaxR_DLt_py_test_40,  MaxR_DLt_py_test_tflite_40, ...
                                MaxR_DLt_py_test_60,  MaxR_DLt_py_test_tflite_60, ...
                                MaxR_DLt_py_test_80,  MaxR_DLt_py_test_tflite_80, ...
                                MaxR_DLt_py_test_100, MaxR_DLt_py_test_tflite_100)
%% Description:


global load_H_files load_Delta_H_max load_DL_dataset load_Rates training save_mat_files load_mat_py;
global seed DeepMIMO_dataset_folder DL_dataset_folder network_folder network_folder_py figure_folder figure_folder_py;

if plot_mode == 2
    MaxR_OPTt_py = MaxR_OPTt_py_test;
    MaxR_OPTt_mat = MaxR_OPTt;
    if epochs == 20
        MaxR_DLt_all = {MaxR_DLt_py_test_20};
        MaxR_DLt_tflite_all = {MaxR_DLt_py_test_tflite_20};
    elseif epochs == 40
        MaxR_DLt_all = {MaxR_DLt_py_test_20, MaxR_DLt_py_test_40};
        MaxR_DLt_tflite_all = {MaxR_DLt_py_test_tflite_20, MaxR_DLt_py_test_tflite_40};
    elseif epochs == 60
        MaxR_DLt_all = {MaxR_DLt_py_test_20, MaxR_DLt_py_test_40, MaxR_DLt_py_test_60};
        MaxR_DLt_tflite_all = {MaxR_DLt_py_test_tflite_20, MaxR_DLt_py_test_tflite_40, MaxR_DLt_py_test_tflite_60};
    elseif epochs == 80
        MaxR_DLt_all = {MaxR_DLt_py_test_20, MaxR_DLt_py_test_40, MaxR_DLt_py_test_60, MaxR_DLt_py_test_80};
        MaxR_DLt_tflite_all = {MaxR_DLt_py_test_tflite_20, MaxR_DLt_py_test_tflite_40, MaxR_DLt_py_test_tflite_60, MaxR_DLt_py_test_tflite_80};
    elseif epochs == 100
        MaxR_DLt_all = {MaxR_DLt_py_test_20, MaxR_DLt_py_test_40, MaxR_DLt_py_test_60, MaxR_DLt_py_test_80, MaxR_DLt_py_test_100};
        MaxR_DLt_tflite_all = {MaxR_DLt_py_test_tflite_20, MaxR_DLt_py_test_tflite_40, MaxR_DLt_py_test_tflite_60, MaxR_DLt_py_test_tflite_80, MaxR_DLt_py_test_tflite_100};
    end
elseif plot_mode == 1
    MaxR_OPTt_mat = MaxR_OPTt;
    MaxR_DLt_all = {MaxR_DLt_mat};
end

%                1    2     3     4     5     6      7      8      9      10     11
Training_Size = [2, 2000, 4000, 6000, 8000, 10000, 14000, 18000, 22000, 26000, 30000];
%Training_Size_number=11; % 30000
%Training_Size_number=6; % 10000
Training_Size_dd=Training_Size(Training_Size_number);

filename_figDpyTestAll=strcat(figure_folder_py, 'FigDpyTestAll_tflite', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(M_ar_master), ' ', ''), strrep(num2str(M_ar_master), ' ', ''), '_Mbar', num2str(M_bar), '_', num2str(Training_Size_dd), '_', num2str(epochs), '.png');

th_step=0.5;

Colour_mat = {'k','k';  % rr = 1   
              'k','k'}; % rr = 2   

Colour_py  = {'k','b','r','g','m','c','y';  % rr = 1
              'k','b','r','g','m','c','y'}; % rr = 2

Marker_py  = {'--', '-'}; % Line styles: dashed for OPT, solid for DL
Marker_py_tflite = {'--', '-'}; % Line styles: dashed for OPT, solid for DL

Marker_py_tflite = { '^',  '*',  's',  'p',  'x';  % rr = 1
                     '^',  '*',  's',  'p',  'x'}; % rr = 2

Marker_mat = {'-o', ':o'}; % Line styles: dashed for OPT, solid for DL

    %          b     blue          .     point              -     solid
    %          g     green         o     circle             :     dotted
    %          r     red           x     x-mark             -.    dashdot 
    %          c     cyan          +     plus               --    dashed   
    %          m     magenta       *     star             (none)  no line
    %          y     yellow        s     square
    %          k     black         d     diamond
    %          w     white         v     triangle (down)
    %                              ^     triangle (up)
    %                              <     triangle (left)
    %                              >     triangle (right)
    %                              p     pentagram
    %                              h     hexagram


fD = figure('Name', 'Figure', 'units','pixels', 'Position', [100, 100, 1200, 400]); % typ 800x400
hold on; grid on; box on;
xlabel('Rate Threshold (bps/Hz)','fontsize',10)
ylabel('Coverage (%)','fontsize',10)
set(gca,'FontSize',11)

%for rr=1:1:numel(My_ar)

if M_ar_master == 32
    colour_idx = 1;
    read_idx = 1;
elseif M_ar_master == 64
    colour_idx = 2;
    read_idx = 2;
end

if plot_mode == 2
    title(['Coverage vs Rate Threshold, RIS ', num2str(M_ar_master), 'x', num2str(M_ar_master), ' with ', num2str(M_bar), ' active elements, pyTest model trained with ', num2str(Training_Size_dd), ' samples'],'fontsize',11)
elseif plot_mode == 1
    title(['Coverage vs Rate Threshold, RIS ', num2str(M_ar_master), 'x', num2str(M_ar_master), ' with ', num2str(M_bar), ' active elements, matVal model trained with ', num2str(Training_Size_dd), ' samples'],'fontsize',11)
end

disp(['---> Plotting FigD with M:', num2str(M_ar_master), ', Training_Size:' num2str(Training_Size_dd), '...']);

%x_plot = 1:(Ur_rows(2)-1000);
y_plot = 1:181;

if plot_mode == 2 || plot_mode == 1

    disp(['plot_mode == ', num2str(plot_mode)])

    disp('load User_Location')
    filename_User_Location=strcat(DeepMIMO_dataset_folder, 'User_Location', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(M_ar_master), num2str(M_ar_master), '_Mbar', num2str(M_bar), '.mat');
    load(filename_User_Location);

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
    
    disp('fill sparsity matrix')
    %% Per ogni sample del test set, recupera la sua posizione e il suo rate massimo
    % Se funziona, questo calcolo si può portare dentro a DL_training_4 per semplificare questo script di plot
    %values_DL = single(nan(181,200));
    values_OPT_py = single(nan(181,200));
    values_OPT_mat = single(nan(181,200));
    for b=1:size(ValTest_Ind,1) % 6200 val o 3100 test
        x = User_Location_norm_ValTest(1,b);
        y = User_Location_norm_ValTest(2,b);

        % MaxR è ordinato come YValidation_un quindi come DL_output_un, che a sua volta è ordinato secondo ValTest_Ind
        %[~,VI_sortind] = sort(ValTest_Ind);
        %[~,VI_rev_sortind] = sort(VI_sortind);
        %DL_output_un = DL_output_un(VI_rev_sortind,:);
        %MaxR_DL(b) = squeeze(YValidation_un(1,1,Indmax_DL(b),b));
        % Quindi MaxR è già ordinato secondo ValTest_Ind
        %values_DL(x,y) = MaxR_DLt(rr,Training_Size_number,b); % equivalent to the previous commented line
    
        %x = User_Location_norm_ValTest(1,b);
        %y = User_Location_norm_ValTest(2,b);

        values_OPT_py(x,y) = MaxR_OPTt_py(read_idx,Training_Size_number,b);
        values_OPT_mat(x,y) = MaxR_OPTt_mat(read_idx,Training_Size_number,b);
    end

    % Matrice dei valori
    values_OPT_py_plot = reshape(values_OPT_py, numel(y_plot), []); % 181x200
    values_OPT_mat_plot = reshape(values_OPT_mat, numel(y_plot), []); % 181x200

    %if plot_mode == 2 || plot_mode == 1
    numero_punti = size(ValTest_Ind,1); % 6200 o 3100
    %end




    % Plot OPT_mat curve
    thresholds = single(0:th_step:floor(max(values_OPT_mat_plot(:)+1)));
    punti_coperti_sul_totale_OPT = arrayfun(@(th) sum(values_OPT_mat_plot(:) >= th) / numero_punti * 100, thresholds);
    plot(thresholds, punti_coperti_sul_totale_OPT, [Colour_mat{colour_idx,1} Marker_mat{1}], 'markersize', 4, 'linewidth', 1., 'DisplayName', ['Genie_{mat}, M = ' num2str(M_ar_master) 'x' num2str(M_ar_master)]);

    % Plot DL_mat curves
    values_DL = single(nan(181,200));
    for b=1:size(ValTest_Ind,1)
        x = User_Location_norm_ValTest(1,b);
        y = User_Location_norm_ValTest(2,b);
        values_DL(x,y) = MaxR_DLt_mat(read_idx,Training_Size_number,b);
    end
    values_DL_plot = reshape(values_DL, numel(y_plot), []);
    punti_coperti_sul_totale_DL = arrayfun(@(th) sum(values_DL_plot(:) >= th) / numero_punti * 100, thresholds);

    plot(thresholds, punti_coperti_sul_totale_DL, [Colour_mat{colour_idx,2} Marker_mat{2}], 'markersize', 4, 'linewidth', 1., 'DisplayName', ['DL_{matVal' num2str(20) '}, M = ' num2str(M_ar_master) 'x' num2str(M_ar_master)]);

    for j=1:numel(thresholds)
        disp([num2str(thresholds(j)), ' ', num2str(punti_coperti_sul_totale_DL(j))])
    end



    % Plot OPT_py curve
    thresholds = single(0:th_step:floor(max(values_OPT_py_plot(:)+1)));
    punti_coperti_sul_totale_OPT = arrayfun(@(th) sum(values_OPT_py_plot(:) >= th) / numero_punti * 100, thresholds);
    plot(thresholds, punti_coperti_sul_totale_OPT, [Colour_py{colour_idx,1} Marker_py{1}], 'markersize', 4, 'linewidth', 1., 'DisplayName', ['Genie_{py}, M = ' num2str(M_ar_master) 'x' num2str(M_ar_master)]);

    for j=1:numel(thresholds)
        disp([num2str(thresholds(j)), ' ', num2str(punti_coperti_sul_totale_OPT(j))])
    end

    % Plot DL_py curves
    for i = 1:numel(MaxR_DLt_all)
        disp(['Elemento ', num2str(i)]);
        values_DL = single(nan(181,200));
        for b=1:size(ValTest_Ind,1)
            x = User_Location_norm_ValTest(1,b);
            y = User_Location_norm_ValTest(2,b);
            values_DL(x,y) = MaxR_DLt_all{i}(read_idx,Training_Size_number,b);
        end
        values_DL_plot = reshape(values_DL, numel(y_plot), []);
        punti_coperti_sul_totale_DL = arrayfun(@(th) sum(values_DL_plot(:) >= th) / numero_punti * 100, thresholds);

        %if i == 1
        %    plot(thresholds, punti_coperti_sul_totale_DL, [Colour_mat{colour_idx,2} Marker_mat{2}], 'markersize', 4, 'linewidth', 1., 'DisplayName', ['DL_{matVal' num2str(20) '}, M = ' num2str(My_ar(rr)) 'x' num2str(Mz_ar(rr))]);
        %else
        %    plot(thresholds, punti_coperti_sul_totale_DL, [Colour_py{colour_idx,i} Marker_py{2}], 'markersize', 4, 'linewidth', 1., 'DisplayName', ['DL_{pyTest' num2str(epochs/5*(i-1)) '}, M = ' num2str(My_ar(rr)) 'x' num2str(Mz_ar(rr))]);
        %end
        plot(thresholds, punti_coperti_sul_totale_DL, [Colour_py{colour_idx,i+1} Marker_py{2}], 'markersize', 4, 'linewidth', 1., 'DisplayName', ['DL_{pyTest' num2str(20*i) '}, M = ' num2str(M_ar_master) 'x' num2str(M_ar_master)]);

        for j=1:numel(thresholds)
            disp([num2str(thresholds(j)), ' ', num2str(punti_coperti_sul_totale_DL(j))])
        end
    end

    % Plot DL_py curves tflite
    for i = 1:numel(MaxR_DLt_tflite_all)
        disp(['Elemento ', num2str(i)]);
        values_DL = single(nan(181,200));
        for b=1:size(ValTest_Ind,1)
            x = User_Location_norm_ValTest(1,b);
            y = User_Location_norm_ValTest(2,b);
            values_DL(x,y) = MaxR_DLt_tflite_all{i}(read_idx,Training_Size_number,b);
        end
        values_DL_plot = reshape(values_DL, numel(y_plot), []);
        punti_coperti_sul_totale_DL = arrayfun(@(th) sum(values_DL_plot(:) >= th) / numero_punti * 100, thresholds);

        %if i == 1
        %    plot(thresholds, punti_coperti_sul_totale_DL, [Colour_mat{colour_idx,2} Marker_mat{2}], 'markersize', 4, 'linewidth', 1., 'DisplayName', ['DL_{matVal' num2str(20) '}, M = ' num2str(My_ar(rr)) 'x' num2str(Mz_ar(rr))]);
        %else
        %    plot(thresholds, punti_coperti_sul_totale_DL, [Colour_py{colour_idx,i} Marker_py{2}], 'markersize', 4, 'linewidth', 1., 'DisplayName', ['DL_{pyTest' num2str(epochs/5*(i-1)) '}, M = ' num2str(My_ar(rr)) 'x' num2str(Mz_ar(rr))]);
        %end
        plot(thresholds, punti_coperti_sul_totale_DL, [Colour_py{colour_idx,i+1} Marker_py_tflite{2}], 'markersize', 4, 'linewidth', 1., 'DisplayName', ['DL_{pyTestTFLite' num2str(20*i) '}, M = ' num2str(M_ar_master) 'x' num2str(M_ar_master)]);

        for j=1:numel(thresholds)
            disp([num2str(thresholds(j)), ' ', num2str(punti_coperti_sul_totale_DL(j))])
        end
    end

    
end
%end

%if numel(My_ar) == 1
lgd = legend('Location','northeast');
%elseif numel(My_ar) == 2
%    lgd = legend('Location','East','NumColumns', 2);
%end
lgd.ItemTokenSize = [20, 9]; % Per ridurre la dimensione dei simboli in legenda. Default [30, 18].
fontsize(lgd,8,'points')
legend show
ylim([0 102]);

%% Save fig
if plot_mode == 2
    saveas(fD, filename_figDpyTestAll);
elseif plot_mode == 1
    saveas(fD, filename_figDmatValAll);
end
close(fD);

disp('Done');