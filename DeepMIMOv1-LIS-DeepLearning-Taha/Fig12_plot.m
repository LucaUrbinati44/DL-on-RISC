function []=Fig12_plot(output_folder,seed,M_bar,Ur_rows,Training_Size,Rate_OPTt,Rate_DLt)
    
    disp(['Plotting Fig12...']);  

    Colour = 'brgmcky';

    f12 = figure('Name', 'Figure12', 'units','pixels');
    hold on; grid on; box on;
    title(['Data Scaling Curve with ' num2str(M_bar) ' active elements'],'fontsize',12)
    xlabel('Deep Learning Training Dataset Size (Thousands of Samples)','fontsize',14)
    ylabel('Achievable Rate (bps/Hz)','fontsize',14)
    set(gca,'FontSize',13)
    if ishandle(f12)
        set(0, 'CurrentFigure', f12)
        hold on; grid on;
        for rr=1:1:numel(Training_Size)
            plot((Training_Size*1e-3),Rate_OPTt(rr,:),[Colour(rr) '*--'],'markersize',8,'linewidth',2, 'DisplayName',['Genie-Aided Reflection Beamforming, M = ' num2str(My_ar(rr)) '*' num2str(Mz_ar(rr))])
            plot((Training_Size*1e-3),Rate_DLt(rr,:),[Colour(rr) 's-'],'markersize',8,'linewidth',2, 'DisplayName', ['DL Reflection Beamforming, M = ' num2str(My_ar(rr)) '*' num2str(Mz_ar(rr))])
        end
        %legend('Location','SouthEast')
        legend('Location','NorthWest')
        legend show
        ylim([0 3]);
    end
    drawnow
    hold off

    sfile_DeepMIMO=strcat(figure_folder, 'Fig12', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar), num2str(Mz_ar), '_Mbar', num2str(M_bar), '_', num2str(numel(Training_Size)), '.png');
    saveas(f12, sfile_DeepMIMO); % Save the figure to a file 
    close(f12); % Close the figure drawnow hold off

    sfile_DeepMIMO=strcat(figure_folder, 'Fig12data', '_seed', num2str(seed), '_grid', num2str(Ur_rows(2)), '_M', num2str(My_ar), num2str(Mz_ar), '_Mbar', num2str(M_bar), '_', num2str(numel(Training_Size)), '.mat');
    save(sfile_DeepMIMO, 'L', 'My_ar', 'Mz_ar', 'M_bar', 'Training_Size', 'K_DL', 'Rate_DLt', 'Rate_OPTt');
