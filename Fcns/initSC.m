function [ssi,ati,sti,st_std,tri,logE,m]=initSC(Trs,ntrials,Tguess,nBins,b,p_spike,A,ef,tau,Dt,offsets,rescalings,a_min)
%initialize spikes and trace
ssi = []; %initial set of shared spike times has no spikes - this will not be sorted but that shouldn't be a problem
ati = cell(ntrials,1); % array of lists of individual trial spike times
sti = cell(ntrials,1); % array of lists of individual trial spike times
sti_ = cell(ntrials,1); % array of lists of individual trial spike times
tri = cell(ntrials,1);
for ti = 1:ntrials
    tri{ti} = b*ones(1,nBins(ti)); %initial trace is set to baseline 
end

N = length(ssi); %number of spikes in spiketrain

%initial logE - compute likelihood initially completely - updates to likelihood will be local
logE = zeros(ntrials,1);
for ti = 1:ntrials
    logE(ti) = sum(-(tri{ti}-Trs{ti}).^2); 
end

m = p_spike*nBins;

logE_ = logE;
for i = 1:length(Tguess)        
    tmpi = Tguess(i); 
    ssi_ = [ssi tmpi];
    %must add spike to each trial (at mean location or sampled -- more appropriate if sampled)
    trti_ = tri;
    ati_ = ati;
    for ti = 1:ntrials
        a_init = max(Trs{ti}(tmpi)/A,a_min);
        tmpti = tmpi +.05*randn;
        [si_, tri_, logE_(ti)] = addSpike(sti{ti},tri{ti},logE_(ti),ef,a_init,tau,Trs{ti},tmpti, N+1, Dt, A, (tmpti-offsets(ti))*rescalings(ti)); %adds all trials' spikes at same time
        sti_{ti} = si_;
        trti_{ti} = tri_;
        ati_{ti} = [ati_{ti} a_init];
    end
    ati = ati_;
    ssi = ssi_;
    sti = sti_;
    tri = trti_;
    N = length(ssi); %number of spikes in spiketrain
end
logE = logE_;

st_std = .1*ones(1,N);
