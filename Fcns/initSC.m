function [ssi,ati,sti,st_std,ci,logC,m]=initSC(CaF,ntrials,Tguess,nBins,b,p_spike,A,ef,tau,Dt,offsets,rescalings,a_min)
%initialize spikes and calcium
ssi = []; %initial set of shared spike times has no spikes - this will not be sorted but that shouldn't be a problem
ati = cell(ntrials,1); % array of lists of individual trial spike times
sti = cell(ntrials,1); % array of lists of individual trial spike times
sti_ = cell(ntrials,1); % array of lists of individual trial spike times
ci = cell(ntrials,1);
for ti = 1:ntrials
    ci{ti} = b*ones(1,nBins(ti)); %initial calcium is set to baseline 
end

N = length(ssi); %number of spikes in spiketrain

%initial logC - compute likelihood initially completely - updates to likelihood will be local
logC = zeros(ntrials,1);
for ti = 1:ntrials
    logC(ti) = sum(-(ci{ti}-CaF{ti}).^2); 
end

m = p_spike*nBins;

logC_ = logC;
for i = 1:length(Tguess)        
    tmpi = Tguess(i); 
    ssi_ = [ssi tmpi];
    %must add spike to each trial (at mean location or sampled -- more appropriate if sampled)
    cti_ = ci;
    ati_ = ati;
    for ti = 1:ntrials
        a_init = max(CaF{ti}(tmpi)/A,a_min);
        tmpti = tmpi +.05*randn;
        [si_, ci_, logC_(ti)] = addSpike(sti{ti},ci{ti},logC_(ti),ef,a_init,tau,CaF{ti},tmpti, N+1, Dt, A, (tmpti-offsets(ti))*rescalings(ti)); %adds all trials' spikes at same time
        sti_{ti} = si_;
        cti_{ti} = ci_;
        ati_{ti} = [ati_{ti} a_init];
    end
    ati = ati_;
    ssi = ssi_;
    sti = sti_;
    ci = cti_;
    N = length(ssi); %number of spikes in spiketrain
end
logC = logC_;

st_std = .1*ones(1,N);
