function []=Fig12_plot(Mx,My_ar,Mz_ar,M_bar,Ur_rows,Training_Size,Rate_OPTt,Rate_DLt)
    
    disp('--> Plotting Fig12...');  

    global load_H_files load_Delta_H_max load_DL_dataset load_Rates save_mat_files;
    global seed DeepMIMO_dataset_folder DL_dataset_folder network_folder figure_folder;

    Colour = 'brgmcky';
    Marker = {'--o', '-o'; '--square', '-square'};
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
            plot((Training_Size*1e-3),Rate_OPTt(rr,:),[Colour(rr) Marker{rr, 1}],'markersize',6,'linewidth',1.5, 'DisplayName',['Genie-Aided Reflection Beamforming, M = ' num2str(My_ar(rr)) 'x' num2str(Mz_ar(rr))])
            plot((Training_Size*1e-3),Rate_DLt(rr,:),[Colour(rr) Marker{rr, 2}],'markersize',6,'linewidth',1.5, 'DisplayName', ['DL Reflection Beamforming, M = ' num2str(My_ar(rr)) 'x' num2str(Mz_ar(rr))])
        end
        %legend('Location','SouthEast')
        lgd = legend('Location','East');
        fontsize(lgd,9,'points')
        legend show
        ylim([0 5.3]);
    end
    drawnow
    hold off

    sfile_DeepMIMO=strcat(figure_folder, 'Fig12', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', strrep(num2str(My_ar), ' ', ''), strrep(num2str(Mz_ar), ' ', ''), '_Mbar', num2str(M_bar), '.png');
    saveas(f12, sfile_DeepMIMO); % Save the figure to a file 
    close(f12); % Close the figure drawnow hold off

    %sfile_DeepMIMO=strcat(figure_folder, 'Fig12data', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar), num2str(Mz_ar), '_Mbar', num2str(M_bar), '_', num2str(numel(Training_Size)), '.mat');
    %save(sfile_DeepMIMO, 'L', 'My_ar', 'Mz_ar', 'M_bar', 'Training_Size', 'K_DL', 'Rate_DLt', 'Rate_OPTt');

    disp('Done');
    
    %keyboard;