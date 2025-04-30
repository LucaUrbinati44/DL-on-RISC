function []=Fig12_plot(My_ar,Mz_ar,M_bar,Ur_rows,Training_Size,...
                        epochs, ...
                        Rate_OPTt,Rate_DLt_mat, ...
                        Rate_DLt_py_valOld_20,Rate_DLt_py_valOld_40, ...
                        Rate_DLt_py_val_20,Rate_DLt_py_test_20, ...
                        Rate_DLt_py_val_40,Rate_DLt_py_test_40, ...
                        Rate_DLt_py_test_60, ...
                        Rate_DLt_py_test_80)
    
    disp('--> Plotting Fig12...');  

    global load_H_files load_Delta_H_max load_DL_dataset load_Rates training save_mat_files load_mat_py;
    global seed DeepMIMO_dataset_folder DL_dataset_folder network_folder network_folder_py figure_folder figure_folder_py;

    filename_fig12=strcat(figure_folder, 'Fig12', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '.png');
    filename_fig12_mat=strcat(figure_folder_py, 'Fig12_mat', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '.png');
    filename_fig12_py=strcat(figure_folder, 'Fig12_py', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '.png');
    filename_fig12_py_test=strcat(figure_folder_py, 'Fig12_py_test', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '_', num2str(epochs), '.png');    
    
    Colour = 'brgm';
    Colour_valOld = 'cg';

    %             1 opt      2 dl mat   
    Marker_mat = {'--',       '--o';  % rr = 1   
                  '--',       '--o'}; % rr = 2

    % dl py valOld     1     2
    Marker_valOld_20_40 = {'--';
                           '--'};

    % dl py val        1      2  
    Marker_val_20_40 = {':';
                        ':'};

    % dl py test       1     2     3     4
    %Marker_test   = {'-s', '-d', '->', '-p';  % rr = 1
    %                 '-s', '-d', '->', '-p'}; % rr = 2
    Marker_test_20_40 = {'-';
                         '-'};
    Marker_test   = {'-^', '-*', '-s', '-p';  % rr = 1
                     '-^', '-*', '-s', '-p'}; % rr = 2

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

    f12 = figure('Name', 'Figure12', 'units','pixels', 'Position', [100, 100, 800, 1200]); % default 600x800
    hold on; grid on; box on;
    title(['Data Scaling Curve with ' num2str(M_bar) ' active elements'],'fontsize',11)
    xlabel('Deep Learning Training Dataset Size (Thousands of Samples)','fontsize',10)
    ylabel('Achievable Rate (bps/Hz)','fontsize',10)
    set(gca,'FontSize',10)
    if ishandle(f12)
        set(0, 'CurrentFigure', f12)
        hold on; grid on;
        for rr=1:1:numel(My_ar)
            plot((Training_Size*1e-3),Rate_OPTt(rr,:),   [Colour(rr) Marker_mat{rr, 1}],'markersize',7,'linewidth',1.2, 'DisplayName', ['Genie, M = ' num2str(My_ar(rr))])
            plot((Training_Size*1e-3),Rate_DLt_mat(rr,:),[Colour(rr) Marker_mat{rr, 2}],'markersize',7,'linewidth',1., 'DisplayName', ['DL_{matVal20}, M = ' num2str(My_ar(rr))])
            disp('Rate_OPTt(rr,:)')
            disp(Rate_OPTt(rr,:))
            disp('Rate_DLt_mat(rr,:)')
            disp(Rate_DLt_mat(rr,:))
            
            if epochs == 20
                plot((Training_Size*1e-3),Rate_DLt_py_valOld_20(rr,:),[Colour_valOld(rr) Marker_valOld_20_40{rr, 1}],'markersize',7,'linewidth',1.2, 'DisplayName', ['DL_{pyValOld20}, M = ' num2str(My_ar(rr))])
                plot((Training_Size*1e-3),Rate_DLt_py_val_20(rr,:),   [Colour(rr) Marker_val_20_40{rr, 1}],   'markersize',7,'linewidth',1.2, 'DisplayName', ['DL_{pyVal20   }, M = ' num2str(My_ar(rr))])
                plot((Training_Size*1e-3),Rate_DLt_py_test_20(rr,:),  [Colour(rr) Marker_test_20_40{rr, 1}],  'markersize',7,'linewidth',1., 'DisplayName', ['DL_{pyTest20  }, M = ' num2str(My_ar(rr))])
                disp('Rate_DLt_py_valOld_20(rr,:)')
                disp(Rate_DLt_py_valOld_20(rr,:))
                disp('Rate_DLt_py_val_20(rr,:)')
                disp(Rate_DLt_py_val_20(rr,:))
                disp('Rate_DLt_py_test_20(rr,:)')
                disp(Rate_DLt_py_test_20(rr,:))            
            elseif epochs == 40
                plot((Training_Size*1e-3),Rate_DLt_py_valOld_40(rr,:),[Colour_valOld(rr) Marker_valOld_20_40{rr, 1}],'markersize',7,'linewidth',1.2, 'DisplayName', ['DL_{pyValOld40}, M = ' num2str(My_ar(rr))])
                plot((Training_Size*1e-3),Rate_DLt_py_val_40(rr,:),   [Colour(rr) Marker_val_20_40{rr, 1}],   'markersize',7,'linewidth',1.2, 'DisplayName', ['DL_{pyVal40   }, M = ' num2str(My_ar(rr))])
                plot((Training_Size*1e-3),Rate_DLt_py_test_40(rr,:),  [Colour(rr) Marker_test_20_40{rr, 1}],  'markersize',7,'linewidth',1., 'DisplayName', ['DL_{pyTest40  }, M = ' num2str(My_ar(rr))])
                disp('Rate_DLt_py_valOld_40(rr,:)')
                disp(Rate_DLt_py_valOld_40(rr,:))
                disp('Rate_DLt_py_val_40(rr,:)')
                disp(Rate_DLt_py_val_40(rr,:))
                disp('Rate_DLt_py_test_40(rr,:)')
                disp(Rate_DLt_py_test_40(rr,:)) 
            elseif epochs == 60
                plot((Training_Size*1e-3),Rate_DLt_py_test_20(rr,:),[Colour(rr) Marker_test{rr, 1}],'markersize',7,'linewidth',1., 'DisplayName', ['DL_{pyTest20}, M = ' num2str(My_ar(rr))])
                plot((Training_Size*1e-3),Rate_DLt_py_test_40(rr,:),[Colour(rr) Marker_test{rr, 2}],'markersize',7,'linewidth',1., 'DisplayName', ['DL_{pyTest40}, M = ' num2str(My_ar(rr))])
                plot((Training_Size*1e-3),Rate_DLt_py_test_60(rr,:),[Colour(rr) Marker_test{rr, 3}],'markersize',7,'linewidth',1., 'DisplayName', ['DL_{pyTest60}, M = ' num2str(My_ar(rr))])
                disp('Rate_DLt_py_test_20(rr,:)')
                disp(Rate_DLt_py_test_20(rr,:))     
                disp('Rate_DLt_py_test_40(rr,:)')
                disp(Rate_DLt_py_test_40(rr,:)) 
                disp('Rate_DLt_py_test_60(rr,:)')
                disp(Rate_DLt_py_test_60(rr,:))
            elseif epochs == 80
                plot((Training_Size*1e-3),Rate_DLt_py_test_20(rr,:),[Colour(rr) Marker_test{rr, 1}],'markersize',7,'linewidth',1., 'DisplayName', ['DL_{pyTest20}, M = ' num2str(My_ar(rr))])
                plot((Training_Size*1e-3),Rate_DLt_py_test_40(rr,:),[Colour(rr) Marker_test{rr, 2}],'markersize',7,'linewidth',1., 'DisplayName', ['DL_{pyTest40}, M = ' num2str(My_ar(rr))])
                plot((Training_Size*1e-3),Rate_DLt_py_test_60(rr,:),[Colour(rr) Marker_test{rr, 3}],'markersize',7,'linewidth',1., 'DisplayName', ['DL_{pyTest60}, M = ' num2str(My_ar(rr))])
                plot((Training_Size*1e-3),Rate_DLt_py_test_80(rr,:),[Colour(rr) Marker_test{rr, 4}],'markersize',7,'linewidth',1., 'DisplayName', ['DL_{pyTest80}, M = ' num2str(My_ar(rr))])
                disp('Rate_DLt_py_test_20(rr,:)')
                disp(Rate_DLt_py_test_20(rr,:))     
                disp('Rate_DLt_py_test_40(rr,:)')
                disp(Rate_DLt_py_test_40(rr,:)) 
                disp('Rate_DLt_py_test_60(rr,:)')
                disp(Rate_DLt_py_test_60(rr,:))
                disp('Rate_DLt_py_test_80(rr,:)')
                disp(Rate_DLt_py_test_80(rr,:))
            end            
        end
        %legend('Location','SouthEast')
        lgd = legend('Location','East','NumColumns', 2);
        %lgd = legend('Location','East');
        lgd.ItemTokenSize = [20, 9]; % Per ridurre la dimensione dei simboli in legenda. Default [30, 18].
        fontsize(lgd,8,'points')

        %lgd.Position = [0.8, 0.2, 0.1, 0.1]; % [x, y, width, height] in coordinate normalizzate

        legend show
        ylim([0 5.3]);
    end
    drawnow
    hold off

    if load_mat_py == 3
        saveas(f12, filename_fig12_py_test);
    elseif load_mat_py == 2
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