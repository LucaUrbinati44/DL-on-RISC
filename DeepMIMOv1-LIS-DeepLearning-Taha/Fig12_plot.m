function []=Fig12_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,Training_Size,Rate_OPTt,Rate_DLt_mat,Rate_DLt_py_20,Rate_DLt_py_40)
    
    disp('--> Plotting Fig12...');  

    global load_H_files load_Delta_H_max load_DL_dataset load_Rates training save_mat_files load_mat_py;
    global seed DeepMIMO_dataset_folder DL_dataset_folder network_folder network_folder_py figure_folder figure_folder_py;

    filename_fig12=strcat(figure_folder, 'Fig12', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '.png');
    filename_fig12_mat=strcat(figure_folder_py, 'Fig12_mat', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '.png');
    filename_fig12_py=strcat(figure_folder, 'Fig12_py', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '.png');
    
    Colour = 'brgmcky';
    %           1           2          3
    %Marker = {'--o',      '-o',      '->';  % rr = 1
    %          '--square', '-square', '-pentagram'}; % rr = 2
    %Marker = {'--',       '-o',      '->';  % rr = 1
    %          '--',       '-square', '-pentagram'}; % rr = 2

    %          1 opt      2 dl mat     3 dl py 20    4 dl py 40
    Marker = {'--',       '-o',       '-square',     '->';  % rr = 1
              '--',       '-o',       '-square',     '->'}; % rr = 2
    % https://uk.mathworks.com/help/matlab/creating_plots/create-line-plot-with-markers.html#bvcbmlz-1
    %disp(size(Marker));

    f12 = figure('Name', 'Figure12', 'units','pixels');
    hold on; grid on; box on;
    title(['Data Scaling Curve with ' num2str(M_bar) ' active elements'],'fontsize',11)
    xlabel('Deep Learning Training Dataset Size (Thousands of Samples)','fontsize',10)
    ylabel('Achievable Rate (bps/Hz)','fontsize',10)
    set(gca,'FontSize',10)
    if ishandle(f12)
        set(0, 'CurrentFigure', f12)
        hold on; grid on;
        for rr=1:1:numel(My_ar)
            plot((Training_Size*1e-3),Rate_OPTt(rr,:),[Colour(rr) Marker{rr, 1}],'markersize',7,'linewidth',1.2, 'DisplayName',      ['Genie,  M = ' num2str(My_ar(rr))])
            plot((Training_Size*1e-3),Rate_DLt_mat(rr,:),[Colour(rr) Marker{rr, 2}],'markersize',7,'linewidth',1.2, 'DisplayName',   ['DL_{mat  }, M = ' num2str(My_ar(rr))])
            plot((Training_Size*1e-3),Rate_DLt_py_20(rr,:),[Colour(rr) Marker{rr, 3}],'markersize',7,'linewidth',1.2, 'DisplayName', ['DL_{py20}, M = ' num2str(My_ar(rr))])
            plot((Training_Size*1e-3),Rate_DLt_py_40(rr,:),[Colour(rr) Marker{rr, 4}],'markersize',7,'linewidth',1.2, 'DisplayName', ['DL_{py40}, M = ' num2str(My_ar(rr))])
            disp('Rate_OPTt(rr,:)')
            disp(Rate_OPTt(rr,:))
            disp('Rate_DLt_mat(rr,:)')
            disp(Rate_DLt_mat(rr,:))
            disp('Rate_DLt_py_20(rr,:)')
            disp(Rate_DLt_py_20(rr,:))
            disp('Rate_DLt_py_40(rr,:)')
            disp(Rate_DLt_py_40(rr,:))
        end
        %legend('Location','SouthEast')
        lgd = legend('Location','East','NumColumns', 2);
        lgd.ItemTokenSize = [20, 9]; % Per ridurre la dimensione dei simboli in legenda. Default [30, 18].
        fontsize(lgd,8,'points')

        %lgd.Position = [0.8, 0.2, 0.1, 0.1]; % [x, y, width, height] in coordinate normalizzate

        legend show
        ylim([0 5.3]);
    end
    drawnow
    hold off

    if load_mat_py == 2
        saveas(f12, filename_fig12_py);
    elseif load_mat_py == 1
        saveas(f12, filename_fig12_mat);
    else
        saveas(f12, filename_fig12);
    end

    close(f12); % Close the figure drawnow hold off

    %sfile_DeepMIMO=strcat(figure_folder, 'Fig12data', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar), num2str(Mz_ar), '_Mbar', num2str(M_bar), '_', num2str(numel(Training_Size)), '.mat');
    %save(sfile_DeepMIMO, 'L', 'My_ar', 'Mz_ar', 'M_bar', 'Training_Size', 'K_DL', 'Rate_DLt', 'Rate_OPTt');

    disp('Done');
    
    %keyboard;