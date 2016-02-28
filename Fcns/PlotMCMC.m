function PlotMCMC(Trs,trials,mcmc,time)
% Once you have your data, then go ahead and plot it
%% multiple samples for same trace
figure(1)
burnIn = round(1/5*length(trials.tau));
ntrials = size(Trs,1); 
final_tr_var = zeros(1,ntrials);
nBins = cellfun(@length,Trs);
for ti = 1:ntrials
    %ML - DOF, init, baseline and each burst amplitude
    SSE=sum((Trs{ti}-trials.curves{end}{ti}).^2);
    n=numel(trials.curves{end}{ti})-(2+mcmc.N_sto(end)); 
    final_tr_var(ti) = SSE/n;%standard error
end
%
final_tr_std = sqrt(final_tr_var);
figure('units','normalized','outerposition',[0 0 .8 .8])
for traceInd=1:min(20,ntrials)
    subplot(min(4,max(floor(ntrials/4),1)),4,traceInd)
    t=time(traceInd,:);
    plot(t(~isnan(t)),Trs{traceInd},'ko'); 
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
    %posterior uncertainty, which is the sum of uncertainty in
    %trace + inherent noise of trace
    plot(t,MTm+2*(final_tr_std(traceInd)+MTs),'r--'); 
    plot(t,MTm-2*(final_tr_std(traceInd)+MTs),'r--'); 
    hold off
    axis tight
    hold off
    axis tight
end

