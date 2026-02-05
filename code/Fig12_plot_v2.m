function []=Fig12_plot(My_ar,Mz_ar,M_bar,Ur_rows,Training_Size,...
                        epochs, ...
                        Rate_OPTt,Rate_DLt_mat, ...
                        Rate_DLt_py_valOld_20,Rate_DLt_py_valOld_40, ...
                        Rate_DLt_py_val_20,Rate_DLt_py_test_20,Rate_DLt_py_test_tflite_20)
    
    disp('--> Plotting Fig12...');  

    global load_mat_py;
    global seed figure_folder figure_folder_py;

    filename_fig12=strcat(figure_folder, 'Fig12', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '.pdf');
    filename_fig12_mat=strcat(figure_folder_py, 'Fig12_mat', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '.pdf');
    filename_fig12_py=strcat(figure_folder, 'Fig12_py', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '.pdf');
    filename_fig12_py_test=strcat(figure_folder_py, 'Fig12_py_test', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '_', num2str(epochs), '.pdf');    
    filename_fig12_py_test_tflite=strcat(figure_folder_py, 'Fig12_py_test_tflite', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '_', num2str(epochs), '.pdf');    
    
    Colour = 'brgm';
    Colour_tflite = 'mg';
    Colour_valOld = 'cg';

    %             1 opt      2 dl mat   
    Marker_mat = {'--',       '-o';  % rr = 1   
                  '--',       '-o'}; % rr = 2

    % dl py valOld     1     2
    Marker_valOld_20_40 = {'--';
                           '--'};

    % dl py val        1      2  
    Marker_val_20_40 = {':';
                        ':'};

    % dl py test       1     2     3     4
    %Marker_test   = {'-s', '-d', '->', '-p';  % rr = 1
    %                 '-s', '-d', '->', '-p'}; % rr = 2
    Marker_test_20_40 = {'-', ':^';
                         '-', ':^'};
    Marker_test   = {'-^', '-*', '-s', '-p', '-x';  % rr = 1
                     '-^', '-*', '-s', '-p', '-x'}; % rr = 2
    %Marker_test   = { '^',  '*',  's',  'p',  'x';  % rr = 1
    %                  '^',  '*',  's',  'p',  'x'}; % rr = 2

    Marker_test_tflite = { '--^',  '--*',  '--s',  '--p',  '--x';  % rr = 1
                           '--^',  '--*',  '--s',  '--p',  '--x'}; % rr = 2
    Marker_test_tflite = { '^',  '*',  's',  'p',  'x';  % rr = 1
                           '^',  '*',  's',  'p',  'x'}; % rr = 2

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

    f12 = figure('Name', 'Figure12', 'units','pixels', 'Position', [100, 100, 500, 400]); % default 600x800
    hold on; grid on; box on;
    set(gca,'FontSize',11)
    %title(['Data Scaling Curve with ' num2str(M_bar) ' active elements'],'fontsize',11)
    xlabel('Training Dataset Size (x1000)','fontsize',11)
    %ylabel('$\widehat{R}_{\mathrm{plot}}$ (Predicted Achievable Rate) [bps/Hz]','fontsize',11, 'Interpreter', 'latex')
    ylabel('$\widehat{R}_{\mathrm{plot}}$ [bps/Hz]','fontsize',13, 'Interpreter', 'latex')
    if ishandle(f12)
        set(0, 'CurrentFigure', f12)
        hold on; grid on;
        for rr=numel(My_ar):-1:1
            mask_valid = ~isnan(Rate_OPTt(rr,:));
            Rate_OPTt_valid = Rate_OPTt(rr,mask_valid);
            Rate_DLt_mat_valid = Rate_DLt_mat(rr,mask_valid);
            Training_Size_valid = Training_Size(mask_valid);
            Rate_DLt_py_test_20_valid = Rate_DLt_py_test_20(rr,mask_valid);
            %disp('mask_valid')
            %disp(mask_valid);
            %disp('Rate_OPTt_valid')
            %disp(Rate_OPTt_valid)
            %disp('Rate_DLt_mat_valid')
            %disp(Rate_DLt_mat_valid)
            %disp('Training_Size_valid')
            %disp(Training_Size_valid);
            %plot((Training_Size_valid*1e-3),Rate_OPTt_valid,   [Colour(rr) Marker_mat{rr, 1}],'markersize',7,'linewidth',1.2, 'DisplayName', ['Genie-Aided$,\; M \mathbin{=} ' num2str(My_ar(rr)) '\times' num2str(Mz_ar(rr)) '$'])
            %plot((Training_Size_valid*1e-3),Rate_DLt_mat_valid,[Colour(rr) Marker_mat{rr, 2}],'markersize',7,'linewidth',1., 'DisplayName', ['$\mathrm{DL}_{\mathrm{val6400}},\; M \mathbin{=} ' num2str(My_ar(rr)) '\times' num2str(Mz_ar(rr)) ',\; \overline{M} \mathbin{=} 8$'])
            plot((Training_Size_valid*1e-3),Rate_OPTt_valid,   [Colour(rr) Marker_mat{rr, 1}], 'markersize',7,'linewidth',1.2, 'DisplayName', ['Genie-Aided$\,,\; M \mathbin{=} ' num2str(My_ar(rr)) ' \,\times\, ' num2str(Mz_ar(rr)) '$']);
            plot((Training_Size_valid*1e-3),Rate_DLt_mat_valid,[Colour(rr) Marker_mat{rr, 2}], 'markersize',7,'linewidth',1., 'DisplayName', ['$\mathrm{DL}_{\mathrm{val6400}} \mathrm{[8]},\; M \mathbin{=} ' num2str(My_ar(rr)) ' \,\times\, ' num2str(Mz_ar(rr)) ',\; \overline{M} \mathbin{=} 8$']);
            
            
            if epochs == 20
                disp("entro")
                %plot((Training_Size*1e-3),Rate_DLt_py_valOld_20(rr,:),      [Colour_valOld(rr) Marker_valOld_20_40{rr, 1}], 'markersize',7,'linewidth',1.2, 'DisplayName', ['DL_{pyValOld20    }, M = ' num2str(My_ar(rr))])
                %plot((Training_Size*1e-3),Rate_DLt_py_val_20(rr,:),         [Colour(rr) Marker_val_20_40{rr, 1}],           'markersize',7,'linewidth',1.2, 'DisplayName', ['DL_{pyVal20       }, M = ' num2str(My_ar(rr))])
                plot((Training_Size_valid*1e-3),Rate_DLt_py_test_20_valid,        [Colour(rr) Marker_test{rr, 1}],          'markersize',7,'linewidth',1.0, 'DisplayName', ['$\mathrm{DL}_{\mathrm{test}      } \mathrm{[ours]},\; M \mathbin{=} ' num2str(My_ar(rr)) ' \,\times\, ' num2str(Mz_ar(rr)) ',\; \overline{M} \mathbin{=} 8$' ])
                %plot((Training_Size*1e-3),Rate_DLt_py_test_tflite_20(rr,:), [Colour(rr) Marker_test_20_40{rr, 2}],          'markersize',7,'linewidth',1.0, 'DisplayName', ['DL_{pyTestTFLite20}, M = ' num2str(My_ar(rr))])
                %disp('Rate_DLt_py_valOld_20(rr,:)')
                %disp(Rate_DLt_py_valOld_20(rr,:))
                %disp('Rate_DLt_py_val_20(rr,:)')
                %disp(Rate_DLt_py_val_20(rr,:))
                disp('Rate_DLt_py_test_20(rr,:)')
                disp(Rate_DLt_py_test_20(rr,:))            
                disp(Rate_DLt_py_test_20_valid)            
                %disp('Rate_DLt_py_test_tflite_20(rr,:)')
                %disp(Rate_DLt_py_test_tflite_20(rr,:))            
            end
        end
        %legend('Location','SouthEast')
        lgd = legend('Location','East','NumColumns', 1);
        lgd.Position = [0.61, 0.445, 0.15, 0.2]; % [x, y, width, height]
        %x: distanza da sinistra (0=sinistra, 1=destra)
        %y: distanza dal basso (0=basso, 1=alto)
        %width, height: dimensioni (puoi lasciarle come quelle di default, oppure ridurle)
        set(lgd, 'Interpreter', 'latex');
        %lgd = legend('Location','East');
        lgd.ItemTokenSize = [20, 9]; % Per ridurre la dimensione dei simboli in legenda. Default [30, 18].
        fontsize(lgd,9,'points')

        %lgd.Position = [0.8, 0.2, 0.1, 0.1]; % [x, y, width, height] in coordinate normalizzate

        legend show
        ylim([0 5.3]);
    end
    drawnow
    hold off

    if load_mat_py == 4
        %saveas(f12, filename_fig12_py_test_tflite);
        set(f12, 'PaperPositionMode', 'auto');
        exportgraphics(f12, filename_fig12_py_test_tflite, 'ContentType', 'vector', 'BackgroundColor', 'none', 'Resolution', 300);
    elseif load_mat_py == 3
        set(f12, 'PaperPositionMode', 'auto');
        exportgraphics(f12, filename_fig12_py_test, 'ContentType', 'vector', 'BackgroundColor', 'none', 'Resolution', 300);
    elseif load_mat_py == 2
        set(f12, 'PaperPositionMode', 'auto');
        exportgraphics(f12, filename_fig12_py, 'ContentType', 'vector', 'BackgroundColor', 'none', 'Resolution', 300);
    elseif load_mat_py == 1
        set(f12, 'PaperPositionMode', 'auto');
        exportgraphics(f12, filename_fig12_mat, 'ContentType', 'vector', 'BackgroundColor', 'none', 'Resolution', 300);
    else
        set(f12, 'PaperPositionMode', 'auto');
        exportgraphics(f12, filename_fig12, 'ContentType', 'vector', 'BackgroundColor', 'none', 'Resolution', 300);
    end

    close(f12); % Close the figure drawnow hold off

    %sfile_DeepMIMO=strcat(figure_folder, 'Fig12data', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar), num2str(Mz_ar), '_Mbar', num2str(M_bar), '_', num2str(numel(Training_Size)), '.mat');
    %save(sfile_DeepMIMO, 'L', 'My_ar', 'Mz_ar', 'M_bar', 'Training_Size', 'K_DL', 'Rate_DLt', 'Rate_OPTt');

    disp('Done');
    
    %keyboard;