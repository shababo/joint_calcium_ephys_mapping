function PlotMCMC(CaF,trials,mcmc,time)
% Once you have your data, then go ahead and plot it
%% multiple samples for same trace
figure(1)
burnIn = round(1/5*length(trials.tau));
ntrials = size(CaF,1); 
final_cal_var = zeros(1,ntrials);
nBins = cellfun(@length,CaF);
for ti = 1:ntrials
    %ML - DOF, init, baseline and each burst amplitude
    SSE=sum((CaF{ti}-trials.curves{end}{ti}).^2);
    n=numel(trials.curves{end}{ti})-(2+mcmc.N_sto(end)); 
    final_cal_var(ti) = SSE/n;%standard error
end
%
final_cal_std = sqrt(final_cal_var);
figure('units','normalized','outerposition',[0 0 .8 .8])
for traceInd=1:min(20,ntrials)
    subplot(min(4,max(floor(ntrials/4),1)),4,traceInd)
    t=time(traceInd,:);
    plot(t(~isnan(t)),CaF{traceInd},'ko'); 
    modelledTrace = [];
    hold on
    for i = burnIn:10:length(trials.curves)
        modelledTrace = [modelledTrace;trials.curves{i}{traceInd}];
        plot(t,trials.curves{i}{traceInd},'r');
        xlabel('time (ms)')
        axis tight
    end
    MTm = mean(modelledTrace);
    MTs = std(modelledTrace);%estimation noise
    plot(t,MTm,'r--');
    %Liams idea: posterior uncertaubnty which is the sum of uncertainty in
    %Ca signal + inherent noise of calcium signal
    plot(t,MTm+2*(final_cal_std(traceInd)+MTs),'r--'); 
    plot(t,MTm-2*(final_cal_std(traceInd)+MTs),'r--'); 
    hold off
    axis tight
    hold off
    axis tight
end
%% For a set of shuffled data, calculate the std on the estimate
% We dont have this
% figure(2);
% %Plot the decrease in std as trial number increases
% convBin2ms = mean(period)*1e3;
% nshuff = 10;
% % subplot(7,2,[11 13])
% all_s_var = [];
% all_ncs = trial_samp{c};
% for k = all_ncs
%     shuff = [];
%     for j = 1:nshuff
% %         load(['shuffles3/realData_indx_' num2str(c) '_nc_' num2str(k) '_shuff_' num2str(j) '.mat'])
%         tmp = convBin2ms*cell2mat(burstTimes(burnIn:end)');
%         shuff = [shuff; (tmp(:,1))];%take just the first spike
%     end
%     all_s_var = [all_s_var std(shuff(:))];
% end
% plot(all_ncs, all_s_var,'k')
% xlabel('number of motifs')
% ylabel('posterior std. of burst time')
% xlim([1 max(trial_samp{c})])
% 
