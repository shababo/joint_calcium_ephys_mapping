function [trials, mcmc, params]  = sampleParams_joint3(traces_c,trace_v,tau_c, tau_v, Tguess, conversion_ratio, params)

%Tguess must be a cell(1,K) where K is the number of neurons
%conversion_ratio: factor to convert calcium timebin to ephys timebin

K = size(traces_c,1);
Tc = size(traces_c,2);
Tv = size(trace_v,2);

%%%%%%%%%%%%%%%%%
% hyperparams
%%%%%%%%%%%%%%%%%
p_spike=params.p_spike;
proposalVar=10;
nsweeps=400; %number of sweeps of sampler
tau1_v_std = 2/20000/params.dt; %proposal variance of tau parameters
tau2_v_std = 2/20000/params.dt; %proposal variance of tau parameters
tau1_c_std = 2/20000/params.dt; %proposal variance of tau parameters
tau2_c_std = 2/20000/params.dt; %proposal variance of tau parameters
tau_v_min = params.tau_v_min;
tau_v_max = params.tau_v_max;
tau_c_min = params.tau_c_min;
tau_c_max = params.tau_c_max;
%all of these are multiplied by big A
Dt=1; %bin unit - don't change this
A=1; % scale factor for all magnitudes for this calcium data setup
% calcium versions
a_c_std = 10; %proposal variance of amplitude
a_c_min = params.a_c_min;
a_c_max = Inf;
b_c_std = 30; %propasal variance of baseline
b_c_min = 0;
b_c_max = 500;
%voltage versions
a_v_std = .2; %proposal variance of amplitude
a_v_min = params.a_v_min;
a_v_max = Inf;
b_v_std = .3; %propasal variance of baseline
b_v_min = 0;
b_v_max = 30;
exclusion_bound = params.eb;%dont let events get within x bins of eachother. this should be in time
nu_0 = 5; %prior on noise - ntrials
sig2_0 = .1; %prior on noise - variance
adddrop = 5;
maxNevents = Inf;

useExtraChannel = 1;

%%%%%%%%%%%%%%%%%
% Calcium params
%%%%%%%%%%%%%%%%

%time constants
nBins_c = length(traces_c); %for all of this, units are bins and spiketrains go from 0 to T where T is number of bins
fBins_c = Tc;
ef_c = genEfilt_ar(tau_c,fBins_c);%exponential filter
ef_c_init = ef_c;
efs_c = cell(1,K);
taus_c = cell(1,K);

%baseline
b_c=median(traces_c,2); %initial baseline value
baseline_c = b_c;

%noise level
cNoiseVar = 10*ones(K,1); 


%%%%%%%%%%%%%%%%%
% ephys params
%%%%%%%%%%%%%%%%

%time constants
nBins_v = length(trace_v); %for all of this, units are bins and spiketrains go from 0 to T where T is number of bins
fBins_v = 5000;
ef_v = genEfilt_ar(tau_v,fBins_v);%exponential filter
ef_v_init = ef_v;

%baseline
b_v=median(trace_v); %initial baseline value
baseline_v = b_v;

%noise level
%noise model is AR(p)
if isfield(params, 'p')
    p = params.p;
else
    p = 2;
end
% phi prior
phi_0 = zeros(p,1);
Phi_0 = 10*eye(p); %inverse covariance 3

vNoiseVar = 10; 


%%%%%%%%%%%%%%%%%
% joint params
%%%%%%%%%%%%%%%%

%delays

%connection strength -- all post-synaptic events from same pre-cell have
%same height and time-constants
efs_v = cell(1,K);
taus_v = cell(1,K);
a_v = zeros(K,1);
a_v_init = max(a_v_min,max(trace_v)/2-baseline_v); %heuristic init for post synaptic event amplitude

%event times
%initialize spikes and calcium
if ~isempty(Tguess)
    Spk = Tguess;
else
    Spk = cell(1,K);
end
a_c = cell(1,K);
phi = [1 zeros(1,p)];



samples_a_v  = cell(1,nsweeps);
samples_a_c  = cell(1,nsweeps);
samples_b_v = cell(1,nsweeps);
samples_b_c = cell(1,nsweeps);
samples_s = cell(1,nsweeps);
samples_v = cell(1,nsweeps);
samples_v_alt = cell(1,nsweeps);
samples_c = cell(1,nsweeps);
samples_tau_v = cell(1,nsweeps);
samples_tau_c = cell(1,nsweeps);
samples_phi = cell(1,nsweeps);
samples_c_noise = cell(1,nsweeps);
samples_v_noise = cell(1,nsweeps);
samples_s_alt = cell(1,nsweeps);
samples_a_alt = cell(1,nsweeps);
samples_tau_alt = cell(1,nsweeps);

N_sto = [];
objective_v = [];
objective_c = [];

% intiailize events and predicted calcium
%this is based on simply what we tell it. 
v = zeros(K,nBins_v); %initial voltage is set to voltage baseline 
c = repmat(b_c,1,nBins_c); %initial voltage is set to voltage baseline 

v_alt = zeros(1,nBins_v); % extra params
ati = []; % array of lists of spike times
sti = []; % array of lists of spike times
taus_alt = cell(1); % array of lists of event taus
efs_alt = cell(1);

N = cellfun(@length,Spk)'; %number of spikes in spiketrain (per neuron observed via calcium)
N_alt = 0;
Nv = sum(N) + N_alt; %total number observed in voltage (could be more than N)

%initial logC - compute likelihood initially completely - updates to likelihood will be local
%for AR(p) noise, we need a different difference inside the likelihood
diff_v = (trace_v-(sum(v) + v_alt + baseline_v)); %trace - prediction

%initial logC - compute likelihood initially completely - updates to likelihood will be local
logC = zeros(K,1);

m = p_spike*nBins_v;


%init calcium & voltage traces

for ki = 1:K
    logC(ki) = sum(-(c(ki,:)-traces_c(ki,:)).^2); 
    
    ssi = Spk{ki}; %these times are in calcium bins
    
    aci = [];
    
    taus_v{ki} = tau_v;
    taus_c{ki} = tau_c;
    efs_v{ki} = ef_v_init;
    efs_c{ki} = ef_c_init;
    a_v(ki) = a_v_init;
    
    if ~isempty(ssi)
        a_init = max(traces_c(ki,round(ssi(1)))/A,a_min);
    end
    for i = 1:length(ssi)   
        %init v, a_v, diff_v
        [~, vi_, diff_v] = addSpike_ar([],v(ki,:),diff_v,ef_v_init,a_v_init,tau_v,trace_v(ki,:),conversion_ratio*ssi(i), i+1, Dt, A, baseline_v + sum(v)-v(ki,:)); 
        v(ki,:) = vi_;        
  
        % init c, a_c, logC
%         a_init = max(traces_c(ki,round(ssi(i)))/A,a_min);
        [~, ci_, logC] = addSpike([],c(ki,:),logC,ef_c_init,a_init,tau_c,traces_c(ki,:),ssi(i), i+1, Dt, A, ssi(i));
        c(ki,:) = ci_;
        aci = [aci a_init];  
    end  
    a_c{ki} = aci; %different amplitude for each calcium event
end
logC_  = logC;

indreport=.1:.1:1;
indreporti=round(nsweeps*indreport);
fprintf('Progress:')



%% loop over sweeps to generate samples
addMoves = [0 0]; %first elem is number successful, second is number total
dropMoves = [0 0];
timeMoves = [0 0];
ampMoves = [0 0];
tauMoves = [0 0];
swapMoves = [0 0];
for i = 1:nsweeps
    
    %%%%%%%%%%%%%%%%%%%%%%%
    % do event time moves
    %%%%%%%%%%%%%%%%%%%%%%%
    for ii = 1:3
        % for each neuron
        for ki = 1:K
            
            ssi = Spk{ki}; %these times are in calcium bins
            
            %for each event
            for ti = 1:length(ssi)  
                
                % select time update
                tmpi = ssi(ti);
                tmpi_ = ssi(ti)+(proposalVar*randn); %add in noise 
                % bouncing off edges
                while tmpi_>nBins_c || tmpi_<0
                    if tmpi_<0
                        tmpi_ = -(tmpi_);
                    elseif tmpi_>nBins_c
                        tmpi_ = nBins_c-(tmpi_-nBins_c);
                    end
                end
                %if its too close to another event, reject this move
                if any(abs(tmpi_-ssi([1:(ti-1) (ti+1):end]))<exclusion_bound)
                    continue
                end
                
                %proposal for calcium
                [ssi_, ci_, logC_(ki)] = removeSpike(ssi,c(ki,:),logC(ki),efs_c{ki},a_c{ki}(ti),taus_c{ki},traces_c(ki,:),tmpi,ti, Dt, A, tmpi);
                [ssi_, ci_, logC_(ki)] = addSpike(ssi_,ci_,logC_(ki),efs_c{ki},a_c{ki}(ti),taus_c{ki},traces_c(ki,:),tmpi_,ti, Dt, A, tmpi_);
                
                %proposal for voltage
                [ssi_b, vi_, diff_v_] = removeSpike_ar(ssi,v(ki,:),diff_v,efs_v{ki},a_v(ki),taus_v{ki},trace_v,conversion_ratio*tmpi,ti, Dt, A, baseline_v + v_alt + sum(v)-v(ki,:));
                [ssi_b, vi_, diff_v_] = addSpike_ar(ssi_b,vi_,diff_v_,efs_v{ki},a_v(ki),taus_v{ki},trace_v,conversion_ratio*tmpi_,ti, Dt, A, baseline_v + v_alt + sum(v)-v(ki,:));
                      
                %accept or reject
                %for prior: (1) use ratio or(2) set prior to 1.
                prior_ratio = 1;

                ratio_v = exp(sum((1/(2*vNoiseVar))*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) )));            
                ratio_c = exp(sum((1/(2*cNoiseVar(ki)))*(logC_(ki)-logC(ki))));
                ratio = ratio_c*ratio_v*prior_ratio;
                if ratio>1 %accept
                    ssi = ssi_;
                    logC(ki) = logC_(ki);
                    c(ki,:) = ci_;
                    v(ki,:) = vi_;
                    diff_v = diff_v_;
                    timeMoves = timeMoves + [1 1];
                    proposalVar = proposalVar + .1*rand*proposalVar/(i);
                elseif rand<ratio %accept
                    ssi = ssi_;
                    logC(ki) = logC_(ki);
                    c(ki,:) = ci_;
                    v(ki,:) = vi_;
                    diff_v = diff_v_;
                    timeMoves = timeMoves + [1 1];
                    proposalVar = proposalVar + .1*rand*proposalVar/(i);
                else
                    %reject - do nothing
                    proposalVar = proposalVar - .1*rand*proposalVar/(i);
                    timeMoves = timeMoves + [0 1];
                end
                                
            end
      
            Spk{ki} = ssi; %these times are in calcium bins
        
        end 
    end
  
    if useExtraChannel
    for ii = 1:3
        %%%%%%%%%%%%%%%%%
        % extra one
        %%%%%%%%%%%%%%%%%
        %for each event
        for ti = 1:N_alt  

            % select time update
            tmpi = sti(ti)/conversion_ratio;
            tmpi_ = tmpi+(proposalVar*randn); %add in noise 
            % bouncing off edges
            while tmpi_>nBins_c || tmpi_<0
                if tmpi_<0
                    tmpi_ = -(tmpi_);
                elseif tmpi_>nBins_c
                    tmpi_ = nBins_c-(tmpi_-nBins_c);
                end
            end
            %if its too close to another event, reject this move
            if any(abs(tmpi_-sti([1:(ti-1) (ti+1):end])/conversion_ratio)<exclusion_bound)
                continue
            end

            %proposal for voltage
            [sti_, vi_, diff_v_] = removeSpike_ar(sti,v_alt,diff_v,efs_alt{ti},ati(ti),taus_alt{ti},trace_v,conversion_ratio*tmpi,ti, Dt, A, baseline_v + sum(v));
            [sti_, vi_, diff_v_] = addSpike_ar(sti_,vi_,diff_v_,efs_alt{ti},ati(ti),taus_alt{ti},trace_v,conversion_ratio*tmpi_,ti, Dt, A, baseline_v + sum(v));
            
            %accept or reject
            %for prior: (1) use ratio or(2) set prior to 1.
            prior_ratio = 1;
            ratio_v = exp(sum((1/(2*vNoiseVar))*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) )));            
            ratio = ratio_v*prior_ratio;
            if ratio>1 %accept
                sti = sti_;
                v_alt = vi_;
                diff_v = diff_v_;
                timeMoves = timeMoves + [1 1];
                proposalVar = proposalVar + .1*rand*proposalVar/(i);
            elseif rand<ratio %accept
                sti = sti_;
                v_alt = vi_;
                diff_v = diff_v_;
                timeMoves = timeMoves + [1 1];
                proposalVar = proposalVar + .1*rand*proposalVar/(i);
            else
                %reject - do nothing
                proposalVar = proposalVar - .1*rand*proposalVar/(i);
                timeMoves = timeMoves + [0 1];
            end

        end        
        
    end
    end
    
    for ii = 1:5
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % update amplitude of each event - calcium
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        % for each neuron
        for ki = 1:K
            
            si = Spk{ki}; %these times are in calcium bins
            if ~isempty(si)
                ai = a_c{ki};

                %sample with random walk proposal
                tmp_a = ai(1);
                tmp_a_ = tmp_a+(a_c_std*randn); %with bouncing off min and max
                while tmp_a_>a_c_max || tmp_a_<a_c_min
                    if tmp_a_<a_c_min
                        tmp_a_ = a_c_min+(a_c_min-tmp_a_);
                    elseif tmp_a_>a_c_max
                        tmp_a_ = a_c_max-(tmp_a_-a_c_max);
                    end
                end
                    
                ai_ = ai;
                
                %for each event
                ci_ = c(ki,:);
                si_ = si;
                logC_(ki) = logC(ki);
                for ti = 1:N(ki) 

                    %all updates (to a channel) use the same modified amplitude 

                    %proposal for calcium
                    [si_, ci_, logC_(ki)] = removeSpike(si_,ci_,logC_(ki),efs_c{ki},ai(ti),taus_c{ki},traces_c(ki,:),si(ti),ti, Dt, A, si(ti));
                    [si_, ci_, logC_(ki)] = addSpike(si_,ci_,logC_(ki),efs_c{ki},tmp_a_,taus_c{ki},traces_c(ki,:),si(ti),ti, Dt, A, si(ti));
                
                    ai_(ti) = tmp_a_;

                end

                %accept or reject - include a prior?
                prior_ratio = 1;
                ratio = exp(sum((1/(2*cNoiseVar(ki)))*(logC_(ki)-logC(ki))))*prior_ratio;
                if ratio>1 %accept
                    c(ki,:) = ci_;
                    logC(ki) = logC_(ki);                    
                    ai = ai_;
                    si = si_;
                    ampMoves = ampMoves + [1 1];
                    a_c_std = a_c_std + 2*.1*rand*a_c_std/(i);
                elseif rand<ratio %accept
                    c(ki,:) = ci_;
                    logC(ki) = logC_(ki);                    
                    ai = ai_;
                    si = si_;
                    ampMoves = ampMoves + [1 1];
                    a_c_std = a_c_std + 2*.1*rand*a_c_std/(i);
                else
                    %reject - do nothing
                    a_c_std = a_c_std - .1*rand*a_c_std/(i);
                    ampMoves = ampMoves + [0 1];
                end
                a_c{ki} = ai;
            end
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % update amplitude of each event - ephys
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % for each neuron
        for ki = 1:K
            
            si = Spk{ki}; %these times are in calcium bins
            if ~isempty(si)
                ai = a_v(ki);

                %for each event -- accept-reject is linked
                diff_v_ = diff_v;
                si_ = si;
                vi_ = v(ki,:);
                
                %sample with random walk proposal
                tmp_a_ = ai+(a_v_std*randn); %with bouncing off min and max
                while tmp_a_>a_v_max || tmp_a_<a_v_min
                    if tmp_a_<a_v_min
                        tmp_a_ = a_v_min+(a_v_min-tmp_a_);
                    elseif tmp_a_>a_v_max
                        tmp_a_ = a_v_max-(tmp_a_-a_v_max);
                    end
                end
                    
                for ti = 1:N(ki) 

                    %proposal for voltage
                    [si_, vi_, diff_v_] = removeSpike_ar(si_,vi_,diff_v_,efs_v{ki},ai,taus_v{ki},trace_v,conversion_ratio*si(ti),ti, Dt, A, baseline_v + v_alt + sum(v)-v(ki,:));
                    [si_, vi_, diff_v_] = addSpike_ar(si_,vi_,diff_v_,efs_v{ki},tmp_a_,taus_v{ki},trace_v,conversion_ratio*si(ti),ti, Dt, A, baseline_v + v_alt + sum(v)-v(ki,:));

                end

                %accept or reject - include a prior?
                prior_ratio = 1;
                ratio = exp(sum((1/(2*vNoiseVar))*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) )))*prior_ratio;
                if ratio>1 %accept
                    v(ki,:) = vi_;
                    diff_v = diff_v_;                    
                    ai = tmp_a_;
                    ampMoves = ampMoves + [1 1];
                    a_v_std = a_v_std + 2*.1*rand*a_v_std/(i);
                elseif rand<ratio %accept
                    v(ki,:) = vi_;
                    diff_v = diff_v_;                    
                    ai = tmp_a_;
                    ampMoves = ampMoves + [1 1];
                    a_v_std = a_v_std + 2*.1*rand*a_v_std/(i);
                else
                    %reject - do nothing
                    a_v_std = a_v_std - .1*rand*a_v_std/(i);
                    ampMoves = ampMoves + [0 1];
                end

                a_v(ki) = ai;
            
            end
        end
        
    end
        if useExtraChannel
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % extra channel
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        si = sti; 
        for ti = 1:length(sti)
            %sample with random walk proposal
            tmp_a = ati(ti);
            tmp_a_ = tmp_a+(a_v_std*randn); %with bouncing off min and max
            while tmp_a_>a_v_max || tmp_a_<a_v_min
                if tmp_a_<a_v_min
                    tmp_a_ = a_v_min+(a_v_min-tmp_a_);
                elseif tmp_a_>a_v_max
                    tmp_a_ = a_v_max-(tmp_a_-a_v_max);
                end
            end

            [si_, vi_, diff_v_] = removeSpike_ar(si,v_alt,diff_v,efs_alt{ti},ati(ti),taus_alt{ti},trace_v,si(ti),ti, Dt, A, baseline_v + sum(v));
            [si_, vi_, diff_v_] = addSpike_ar(si_,vi_,diff_v_,efs_alt{ti},tmp_a_,taus_alt{ti},trace_v,si(ti),ti, Dt, A, baseline_v + sum(v));
            
            %accept or reject - include a prior?
            prior_ratio = 1;
            ratio = exp(sum((1/(2*vNoiseVar))*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) )))*prior_ratio;
            if ratio>1 %accept
                ati(ti) = tmp_a_;
                si = si_;
                v_alt = vi_;
                diff_v = diff_v_;
%                 ampMoves = ampMoves + [1 1];
%                 a_std = a_std + 2*.1*rand*a_std/(i);
            elseif rand<ratio %accept
                ati(ti) = tmp_a_;
                si = si_;
                v_alt = vi_;
                diff_v = diff_v_;
%                 ampMoves = ampMoves + [1 1];
%                 a_std = a_std + 2*.1*rand*a_std/(i);
            else
                %reject - do nothing
%                 a_std = a_std - .1*rand*a_std/(i);
%                 ampMoves = ampMoves + [0 1];
            end
        end
        end         
    
    
    % update baseline of each trial
    for ii = 1:1
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % update baseline of each trace - calcium
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for ki = 1:K
            %sample with random walk proposal
            tmp_b = baseline_c(ki);
            tmp_b_ = tmp_b+(b_c_std*randn); %with bouncing off min and max
            while tmp_b_>b_c_max || tmp_b_<b_c_min
                if tmp_b_<b_c_min
                    tmp_b_ = b_c_min+(b_c_min-tmp_b_);
                elseif tmp_b_>b_c_max
                    tmp_b_ = b_c_max-(tmp_b_-b_c_max);
                end
            end

            [ci_, logC_(ki)] = remove_base(c(ki,:),logC(ki),tmp_b,traces_c(ki,:),A);   
            [ci_, logC_(ki)] = add_base(ci_,logC_(ki),tmp_b_,traces_c(ki,:),A);
                
            %accept or reject - include a prior?
            prior_ratio = 1;
            ratio = exp(sum((1/(2*cNoiseVar(ki)))*(logC_(ki)-logC(ki))))*prior_ratio;
            if ratio>1 %accept
                baseline_c(ki) = tmp_b_;
                c(ki,:) = ci_;
                logC(ki) = logC_(ki);
                b_c_std = b_c_std + 2*.1*rand*b_c_std/(i);
            elseif rand<ratio %accept
                baseline_c(ki) = tmp_b_;
                c(ki,:) = ci_;
                logC(ki) = logC_(ki);
                b_c_std = b_c_std + 2*.1*rand*b_c_std/(i);
            else
                b_c_std = b_c_std - .1*rand*b_c_std/(i);
                %reject - do nothing
            end
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % update baseline of trace - ephys
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %sample with random walk proposal
        tmp_b = baseline_v;
        tmp_b_ = tmp_b+(b_v_std*randn); %with bouncing off min and max
        while tmp_b_>b_v_max || tmp_b_<b_v_min
            if tmp_b_<b_v_min
                tmp_b_ = b_v_min+(b_v_min-tmp_b_);
            elseif tmp_b_>b_v_max
                tmp_b_ = b_v_max-(tmp_b_-b_v_max);
            end
        end
        
        diff_v_ = (trace_v-(sum(v) + v_alt + tmp_b_)); %trace - prediction

        %accept or reject - include a prior?
        prior_ratio = 1;
        ratio = exp(sum((1/(2*vNoiseVar))*(  predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1)  )))*prior_ratio;
        if ratio>1 %accept
            baseline_v = tmp_b_;
            diff_v = diff_v_;
            b_v_std = b_v_std + 2*.1*rand*b_v_std/(i);
        elseif rand<ratio %accept
            baseline_v = tmp_b_;
            diff_v = diff_v_;
            b_v_std = b_v_std + 2*.1*rand*b_v_std/(i);
        else
            b_v_std = b_v_std - .1*rand*b_v_std/(i);
            %reject - do nothing
        end
        
        
    end
    
    
    
    
    
    

    if i>0
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % updates the number of spikes (add/drop)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for ki = 1:K
            for ii = 1:adddrop 
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Add
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                %propose a uniform add
                %pick a random point
                tmpi = min(nBins_c)*rand;
                %dont add if we have too many events or the proposed new location
                %is too close to another one
                if ~(any(abs(tmpi-Spk{ki})<exclusion_bound) || N(ki) >= maxNevents)                    
                    
                    %proposal for calcium
                    if isempty(Spk{ki})
                        a_init = max(traces_c(ki,max(1,floor(tmpi)))/A - baseline_c(ki) + a_c_std*randn,a_c_min);%propose an initial amplitude for it
                    else
                        a_init = a_c{ki}(1);
                    end
                    [ssi_, ci_, logC_(ki)] = addSpike(Spk{ki},c(ki,:),logC(ki),efs_c{ki},a_init,taus_c{ki},traces_c(ki,:),tmpi,N(ki)+1, Dt, A, tmpi);
                
                    %proposal for voltage
                    [~, vi_, diff_v_] = addSpike_ar(Spk{ki},v(ki,:),diff_v,efs_v{ki},a_v(ki),taus_v{ki},trace_v,conversion_ratio*tmpi,N(ki)+1, Dt, A, baseline_v + v_alt + sum(v)-v(ki,:));

                    %accept or reject     
                    prior_ratio = (m)/(Nv+1); %poisson prior on post-synapse.
                    ratio_v = exp(sum((1/(2*vNoiseVar))*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) )));   
                    ratio_c = exp(sum((1/(2*cNoiseVar(ki)))*(logC_(ki)-logC(ki))));
                    ratio = ratio_c*ratio_v*prior_ratio;
                    if (ratio>1)||(ratio>rand) %accept
                        Spk{ki} = ssi_;
                        c(ki,:) = ci_;
                        logC(ki) = logC_(ki);
                        v(ki,:) = vi_;
                        diff_v = diff_v_;
                        a_c{ki} = [a_c{ki} a_init];
                        addMoves = addMoves + [1 1];
                    else
                        %reject - do nothing
                        addMoves = addMoves + [0 1];
                    end
                    N(ki) = length(Spk{ki});
                    N_alt = length(sti);
                    Nv = sum(N) + N_alt;
                end


                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Drop
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if N(ki)>0%i.e. we if have at least one spike           
                    %propose a uniform removal
                    tmpi = randi(N(ki));%pick one of the spikes at random
                 
                    %proposal for calcium
                    [ssi_, ci_, logC_(ki)] = removeSpike(Spk{ki},c(ki,:),logC(ki),efs_c{ki},a_c{ki}(tmpi),taus_c{ki},traces_c(ki,:),Spk{ki}(tmpi),tmpi, Dt, A, Spk{ki}(tmpi));
                
                    %proposal for voltage
                    [~, vi_, diff_v_] = removeSpike_ar(Spk{ki},v(ki,:),diff_v,efs_v{ki},a_v(ki),taus_v{ki},trace_v,conversion_ratio*Spk{ki}(tmpi),tmpi, Dt, A, baseline_v + v_alt + sum(v)-v(ki,:));

                    %accept or reject     
                    prior_ratio = (m)/(Nv+1); %poisson prior on post-synapse.
                    ratio_v = exp(sum((1/(2*vNoiseVar))*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) )));   
                    ratio_c = exp(sum((1/(2*cNoiseVar(ki)))*(logC_(ki)-logC(ki))));
                    ratio = ratio_c*ratio_v*prior_ratio;
                    if (ratio>1)||(ratio>rand) %accept
                        Spk{ki} = ssi_;
                        c(ki,:) = ci_;
                        logC(ki) = logC_(ki);
                        v(ki,:) = vi_;
                        diff_v = diff_v_;
                        a_c{ki}(tmpi) = [];
                        dropMoves = dropMoves + [1 1];
                    else
                        %reject - do nothing
                        dropMoves = dropMoves + [0 1];
                    end
                    N(ki) = length(Spk{ki});
                    N_alt = length(sti);
                    Nv = sum(N) + N_alt;
                end
                
             end
            
        end  
        if useExtraChannel
        for ii = 1:adddrop 
            %%%%%%%%%%%%%%%%%%%%%%%%
            % extra voltage channel
            %%%%%%%%%%%%%%%%%%%%%%%%
            tmpi = min(nBins_c)*rand;
            if ~(any(abs(tmpi-sti)<exclusion_bound) || N_alt >= maxNevents)
                [si_, vi_, diff_v_] = addSpike_ar(sti,v_alt,diff_v,ef_v_init,a_v_init,tau_v,trace_v,conversion_ratio*tmpi,N_alt+1, Dt, A, baseline_v + sum(v));
                ati_ = [ati a_v_init];
            
                prior_ratio = (m)/(Nv+1); %poisson prior on post-synapse.
                ratio_v = exp(sum((1/(2*vNoiseVar))*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) )));   
                ratio = ratio_v*prior_ratio;                    
                if (ratio>1)||(ratio>rand) %accept
                    ati = ati_;
                    sti = si_;
                    v_alt = vi_;
                    taus_alt{N_alt+1} = tau_v;
                    efs_alt{N_alt+1} = ef_v_init;
                    diff_v = diff_v_;
                else
                    %reject - do nothing
                end
                N_alt = length(sti);
                Nv = sum(N) + N_alt;
            end
        end
        for ii = 1:adddrop
            % delete
            if N_alt>0%i.e. we if have at least one spike           
                %propose a uniform removal
                tmpi = randi(N_alt);%pick one of the spikes at random
                %always remove the ith event (the ith event of each trial is linked)                     
                [si_, vi_, diff_v_] = removeSpike_ar(sti,v_alt,diff_v,efs_alt{tmpi},ati(tmpi),taus_alt{tmpi},trace_v,sti(tmpi),tmpi, Dt, A, baseline_v + sum(v));
            
                %accept or reject
                %posterior times reverse prob/forward prob
                prior_ratio = (m)/(Nv+1); %poisson prior on post-synapse.
                ratio_v = exp(sum((1/(2*vNoiseVar))*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) )));   
                ratio = ratio_v*prior_ratio;   
                if (ratio>1)||(ratio>rand)%accept
                    ati(tmpi) = [];
                    sti = si_;
                    v_alt = vi_;
                    taus_alt(tmpi) = [];
                    efs_alt(tmpi) = [];
                    diff_v = diff_v_;
                else
                    %reject - do nothing
                end
                N_alt = length(sti);
                Nv = sum(N) + N_alt;
            end    
        end
        end
        
    end
    


    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % updates taus for each calcium trace
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% this is the section that updates tau
    % update tau (via random walk sampling)
    for ii = 1:1
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % first tau (rise)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for ki = 1:K 
            if ~isempty(Spk{ki})
                % update both tau values
                tau_ = taus_c{ki};
                tau_(1) = tau_(1)+(tau1_c_std*randn); %with bouncing off min and max
                while tau_(1)>tau_v(2) || tau_(1)<tau_c_min
                    if tau_(1)<tau_c_min
                        tau_(1) = tau_c_min+(tau_c_min-tau_(1));
                    elseif tau_(1)>tau_v(2)
                        tau_(1) = tau_v(2)-(tau_(1)-tau_v(2));
                    end
                end 

                ef_ = genEfilt_ar(tau_,fBins_c);%exponential filter

                %remove all old bumps and replace them with new bumps -- loop over events   
                si = Spk{ki};
                si_ = si;
                ci_ = c(ki,:);
                logC_ki = logC(ki);
                for ti = 1:N(ki)
                    %proposal for calcium
                    [si_, ci_, logC_ki] = removeSpike(si_,ci_,logC_ki,efs_c{ki},a_c{ki}(ti),taus_c{ki},traces_c(ki,:),si(ti),ti, Dt, A, si(ti));
                    [si_, ci_, logC_ki] = addSpike(si_,ci_,logC_ki,ef_,a_c{ki}(ti),tau_,traces_c(ki,:),si(ti),ti, Dt, A, si(ti));
                end
                
                %accept or reject
                prior_ratio = 1;
    %             prior_ratio = (gampdf(tau_(1),1.5,1))/(gampdf(tau(1),1.5,1));
                ratio = exp(sum((1/(2*cNoiseVar(ki)))*(logC_(ki)-logC(ki))))*prior_ratio;
                if ratio>1 %accept
                    c(ki,:) = ci_;
                    logC(ki) = logC_ki;
                    taus_c{ki} = tau_;
                    efs_c{ki} = ef_;
                    tauMoves = tauMoves + [1 1];
                    tau1_c_std = tau1_c_std + .1*rand*tau1_c_std/(i);
                elseif rand<ratio %accept
                    c(ki,:) = ci_;
                    logC(ki) = logC_ki;
                    taus_c{ki} = tau_;
                    efs_c{ki} = ef_;
                    tauMoves = tauMoves + [1 1];
                    tau1_c_std = tau1_c_std + .1*rand*tau1_c_std/(i);
                else
                    %reject - do nothing
                    tau1_c_std = tau1_c_std - .1*rand*tau1_c_std/(i);
                    tauMoves = tauMoves + [0 1];
                end
            end
        end
    
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % second tau (fall)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for ki = 1:K  
            if ~isempty(Spk{ki})
                % update both tau values
                tau_ = taus_c{ki};    
                tau_(2) = tau_(2)+(tau2_c_std*randn);
                while tau_(2)>tau_c_max || tau_(2)<tau_(1)
                    if tau_(2)<tau_(1)
                        tau_(2) = tau_(1)+(tau_(1)-tau_(2));
                    elseif tau_(2)>tau_c_max
                        tau_(2) = tau_c_max-(tau_(2)-tau_c_max);
                    end
                end  

                ef_ = genEfilt_ar(tau_,fBins_c);%exponential filter

                %remove all old bumps and replace them with new bumps -- loop over events   
                si = Spk{ki};
                si_ = si;
                ci_ = c(ki,:);
                logC_ki = logC(ki);
                for ti = 1:N(ki)
                    %proposal for calcium
                    [si_, ci_, logC_ki] = removeSpike(si_,ci_,logC_ki,efs_c{ki},a_c{ki}(ti),taus_c{ki},traces_c(ki,:),si(ti),ti, Dt, A, si(ti));
                    [si_, ci_, logC_ki] = addSpike(si_,ci_,logC_ki,ef_,a_c{ki}(ti),tau_,traces_c(ki,:),si(ti),ti, Dt, A, si(ti));
                end
                
                %accept or reject
                prior_ratio = 1;
    %             prior_ratio = (gampdf(tau_(1),1.5,1))/(gampdf(tau(1),1.5,1));
                ratio = exp(sum((1/(2*cNoiseVar(ki)))*(logC_(ki)-logC(ki))))*prior_ratio;
                if ratio>1 %accept
                    c(ki,:) = ci_;
                    logC(ki) = logC_ki;
                    taus_c{ki} = tau_;
                    efs_c{ki} = ef_;
                    tauMoves = tauMoves + [1 1];
                    tau2_c_std = tau2_c_std + .1*rand*tau2_c_std/(i);
                elseif rand<ratio %accept
                    c(ki,:) = ci_;
                    logC(ki) = logC_ki;
                    taus_c{ki} = tau_;
                    efs_c{ki} = ef_;
                    tauMoves = tauMoves + [1 1];
                    tau2_c_std = tau2_c_std + .1*rand*tau2_c_std/(i);
                else
                    %reject - do nothing
                    tau2_c_std = tau2_c_std - .1*rand*tau2_c_std/(i);
                    tauMoves = tauMoves + [0 1];
                end

            end
        end
        
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % updates taus for each ephys input channel
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % first tau (rise)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for ki = 1:K 
            if ~isempty(Spk{ki})
                % update both tau values
                tau_ = taus_v{ki};
                tau_(1) = tau_(1)+(tau1_v_std*randn); %with bouncing off min and max
                while tau_(1)>tau_(2) || tau_(1)<tau_v_min
                    if tau_(1)<tau_v_min
                        tau_(1) = tau_v_min+(tau_v_min-tau_(1));
                    elseif tau_(1)>tau_(2)
                        tau_(1) = tau_(2)-(tau_(1)-tau_(2));
                    end
                end 

                ef_ = genEfilt_ar(tau_,fBins_v);%exponential filter

                %remove all old bumps and replace them with new bumps    
                si = Spk{ki};
                si_ = si;
                vi_ = v(ki,:);    
                diff_v_ = diff_v;
                for ti = 1:N(ki)
                    %proposal for voltage
                    [si_, vi_, diff_v_] = removeSpike_ar(si_,vi_,diff_v_,efs_v{ki},a_v(ki),taus_v{ki},trace_v,conversion_ratio*si(ti),ti, Dt, A, baseline_v + v_alt + sum(v)-v(ki,:));
                    [si_, vi_, diff_v_] = addSpike_ar(si_,vi_,diff_v_,ef_,a_v(ki),tau_,trace_v,conversion_ratio*si(ti),ti, Dt, A, baseline_v + v_alt + sum(v)-v(ki,:));
                end

                %accept or reject
                prior_ratio = 1;
    %             prior_ratio = (gampdf(tau_(1),1.5,1))/(gampdf(tau(1),1.5,1));
                ratio = exp(sum(sum((1./(2*vNoiseVar)).*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) ))))*prior_ratio;
                if ratio>1 %accept
                    v(ki,:) = vi_;
                    diff_v = diff_v_;
                    taus_v{ki} = tau_;
                    efs_v{ki} = ef_;
                    tauMoves = tauMoves + [1 1];
                    tau1_v_std = tau1_v_std + .1*rand*tau1_v_std/(i);
                elseif rand<ratio %accept
                    v(ki,:) = vi_;
                    diff_v = diff_v_;
                    taus_v{ki} = tau_;
                    efs_v{ki} = ef_;
                    tauMoves = tauMoves + [1 1];
                    tau1_v_std = tau1_v_std + .1*rand*tau1_v_std/(i);
                else
                    %reject - do nothing
                    tau1_v_std = tau1_v_std - .1*rand*tau1_v_std/(i);
                    tauMoves = tauMoves + [0 1];
                end
            end
        end
    
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % second tau (fall)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for ki = 1:K 
            if ~isempty(Spk{ki})
                % update both tau values
                tau_ = taus_v{ki};    
                tau_(2) = tau_(2)+(tau2_v_std*randn);
                while tau_(2)>tau_v_max || tau_(2)<tau_(1)
                    if tau_(2)<tau_(1)
                        tau_(2) = tau_(1)+(tau_(1)-tau_(2));
                    elseif tau_(2)>tau_v_max
                        tau_(2) = tau_v_max-(tau_(2)-tau_v_max);
                    end
                end  

                ef_ = genEfilt_ar(tau_,fBins_v);%exponential filter

                %remove all old bumps and replace them with new bumps    
                si = Spk{ki};
                si_ = si;
                vi_ = v(ki,:);    
                diff_v_ = diff_v;
                for ti = 1:N(ki)
                    %proposal for voltage
                    [si_, vi_, diff_v_] = removeSpike_ar(si_,vi_,diff_v_,efs_v{ki},a_v(ki),taus_v{ki},trace_v,conversion_ratio*si(ti),ti, Dt, A, baseline_v + v_alt + sum(v)-v(ki,:));
                    [si_, vi_, diff_v_] = addSpike_ar(si_,vi_,diff_v_,ef_,a_v(ki),tau_,trace_v,conversion_ratio*si(ti),ti, Dt, A, baseline_v + v_alt + sum(v)-v(ki,:));
                end

                %accept or reject
                prior_ratio = 1;
    %             prior_ratio = (gampdf(tau_(1),1.5,1))/(gampdf(tau(1),1.5,1));
                ratio = exp(sum(sum((1./(2*vNoiseVar)).*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) ))))*prior_ratio;
                if ratio>1 %accept
                    v(ki,:) = vi_;
                    diff_v = diff_v_;
                    taus_v{ki} = tau_;
                    efs_v{ki} = ef_;
                    tauMoves = tauMoves + [1 1];
                    tau2_v_std = tau2_v_std + .1*rand*tau2_v_std/(i);
                elseif rand<ratio %accept
                    v(ki,:) = vi_;
                    diff_v = diff_v_;
                    taus_v{ki} = tau_;
                    efs_v{ki} = ef_;
                    tauMoves = tauMoves + [1 1];
                    tau2_v_std = tau2_v_std + .1*rand*tau2_v_std/(i);
                else
                    %reject - do nothing
                    tau2_v_std = tau2_v_std - .1*rand*tau2_v_std/(i);
                    tauMoves = tauMoves + [0 1];
                end
            end
        end
            
        if useExtraChannel    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % updates taus for extra input channel
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % first tau (rise)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            for ti = 1:N_alt 
                tau_ = taus_alt{ti};
                tau_(1) = tau_(1)+(tau1_v_std*randn); %with bouncing off min and max
                while tau_(1)>tau_(2) || tau_(1)<tau_v_min
                    if tau_(1)<tau_v_min
                        tau_(1) = tau_v_min+(tau_v_min-tau_(1));
                    elseif tau_(1)>tau_(2)
                        tau_(1) = tau_(2)-(tau_(1)-tau_(2));
                    end
                end 
                
                ef_ = genEfilt_ar(tau_,fBins_v);%exponential filter

                %proposal for voltage
                [si_, vi_, diff_v_] = removeSpike_ar(sti,v_alt,diff_v,efs_alt{ti},ati(ti),taus_alt{ti},trace_v,sti(ti),ti, Dt, A, baseline_v + sum(v));
                [si_, vi_, diff_v_] = addSpike_ar(si_,vi_,diff_v_,ef_,ati(ti),tau_,trace_v,sti(ti),ti, Dt, A, baseline_v + sum(v));
                
            
                %accept or reject
                prior_ratio = 1;
    %             prior_ratio = (gampdf(tau_(1),1.5,1))/(gampdf(tau(1),1.5,1));
                ratio = exp(sum(sum((1./(2*vNoiseVar)).*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) ))))*prior_ratio;
                if ratio>1 %accept
                    v_alt = vi_;
                    diff_v = diff_v_;
                    taus_alt{ti} = tau_;
                    efs_alt{ti} = ef_;
                elseif rand<ratio %accept
                    v_alt = vi_;
                    diff_v = diff_v_;
                    taus_alt{ti} = tau_;
                    efs_alt{ti} = ef_;
                else
                    %reject - do nothing
                end
            end
                        
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % second tau (fall)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            for ti = 1:N_alt 
                % update both tau values
                tau_ = taus_alt{ti};       
                tau_(2) = tau_(2)+(tau2_v_std*randn);
                while tau_(2)>tau_v_max || tau_(2)<tau_(1)
                    if tau_(2)<tau_(1)
                        tau_(2) = tau_(1)+(tau_(1)-tau_(2));
                    elseif tau_(2)>tau_v_max
                        tau_(2) = tau_v_max-(tau_(2)-tau_v_max);
                    end
                end  
                
                ef_ = genEfilt_ar(tau_,fBins_v);%exponential filter

                %proposal for voltage
                [si_, vi_, diff_v_] = removeSpike_ar(sti,v_alt,diff_v,efs_alt{ti},ati(ti),taus_alt{ti},trace_v,sti(ti),ti, Dt, A, baseline_v + sum(v));
                [si_, vi_, diff_v_] = addSpike_ar(si_,vi_,diff_v_,ef_,ati(ti),tau_,trace_v,sti(ti),ti, Dt, A, baseline_v + sum(v));
                
            
                %accept or reject
                prior_ratio = 1;
    %             prior_ratio = (gampdf(tau_(1),1.5,1))/(gampdf(tau(1),1.5,1));
                ratio = exp(sum(sum((1./(2*vNoiseVar)).*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) ))))*prior_ratio;
                if ratio>1 %accept
                    v_alt = vi_;
                    diff_v = diff_v_;
                    taus_alt{ti} = tau_;
                    efs_alt{ti} = ef_;
                elseif rand<ratio %accept
                    v_alt = vi_;
                    diff_v = diff_v_;
                    taus_alt{ti} = tau_;
                    efs_alt{ti} = ef_;
                else
                    %reject - do nothing
                end
            end
        end 
    end
    
    
    
    if i>0
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % swap moves -- unidirectional
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for ki = 1:K %the guy to swap from
            kj = randi(K,1);
            if isempty(Spk{ki})
                continue
            else
                tmpi = randi(N(ki));%pick one of the spikes at random
            end

            %try dropping tmpi from ki and adding to kj (accept or reject)\

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Drop (from ki)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %proposal for calcium
            [ssi_i, ci_i, logC_ki] = removeSpike(Spk{ki},c(ki,:),logC(ki),efs_c{ki},a_c{ki}(tmpi),taus_c{ki},traces_c(ki,:),Spk{ki}(tmpi),tmpi, Dt, A, Spk{ki}(tmpi));

            %proposal for voltage
            [~, vi_i, diff_v_] = removeSpike_ar(Spk{ki},v(ki,:),diff_v,efs_v{ki},a_v(ki),taus_v{ki},trace_v,conversion_ratio*Spk{ki}(tmpi),tmpi, Dt, A, baseline_v + v_alt + sum(v)-v(ki,:));

            v_ = v;
            v_(ki,:) = vi_i;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Add (to kj)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

            tmpj = Spk{ki}(tmpi);

            if (any(abs(tmpj-Spk{kj})<exclusion_bound) || N(kj) >= maxNevents)  
                continue
            else
                %proposal for calcium
                if isempty(Spk{kj})
                    a_init = max(traces_c(kj,max(1,floor(tmpj)))/A - baseline_c(kj) + a_c_std*randn,a_c_min);%propose an initial amplitude for it
                else
                    a_init = a_c{kj}(1);
                end
                [ssi_j, ci_j, logC_kj] = addSpike(Spk{kj},c(kj,:),logC(kj),efs_c{kj},a_init,taus_c{kj},traces_c(kj,:),tmpj,N(kj)+1, Dt, A, tmpj);

                %proposal for voltage
                [~, vi_j, diff_v_] = addSpike_ar(Spk{kj},v_(kj,:),diff_v_,efs_v{kj},a_v(kj),taus_v{kj},trace_v,conversion_ratio*tmpj,N(kj)+1, Dt, A, baseline_v + v_alt + sum(v_)-v_(kj,:)); 
                v_(kj,:) = vi_j;
            end


            %accept or reject     
            prior_ratio = 1;
            ratio_v = exp(sum((1/(2*vNoiseVar))*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) )));   
            ratio_c_i = exp(sum((1/(2*cNoiseVar(ki)))*(logC_ki-logC(ki))));
            ratio_c_j = exp(sum((1/(2*cNoiseVar(kj)))*(logC_kj-logC(kj))));
            ratio = ratio_c_i*ratio_c_j*ratio_v*prior_ratio;
            if (ratio>1)||(ratio>rand) %accept
                Spk{ki} = ssi_i;
                Spk{kj} = ssi_j;
                c(ki,:) = ci_i;
                c(kj,:) = ci_j;
                logC(ki) = logC_ki;
                logC(kj) = logC_kj;
                v = v_;
                diff_v = diff_v_;
                a_c{ki}(tmpi) = [];
                a_c{kj} = [a_c{kj} a_init];
                swapMoves = swapMoves + [1 1];
            else
                %reject - do nothing
                swapMoves = swapMoves + [0 1];
            end
            N(ki) = length(Spk{ki});
            N(kj) = length(Spk{kj});
            Nv = sum(N) + N_alt;
        end
        
        
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % swap moves -- bidirectional
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for ki = 1:K %the guy to swap from
            tmp = setdiff(1:K,ki);
            kj = tmp(randi(numel(tmp)));
            if isempty(Spk{ki}) || isempty(Spk{kj})
                continue
            else
                tmpi1 = randi(N(ki));%pick one of the spikes at random
                [~, tmpi2] = min( abs(Spk{ki}(tmpi1)-Spk{kj}) );% pick nearest spike
            end

            %try dropping tmpi from ki and adding to kj (accept or reject)\

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Drop (tmpi1 from ki, tmpi2 from kj)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %proposal for calcium
            [ssi_i, ci_i, logC_ki] = removeSpike(Spk{ki},c(ki,:),logC(ki),efs_c{ki},a_c{ki}(tmpi1),taus_c{ki},traces_c(ki,:),Spk{ki}(tmpi1),tmpi1, Dt, A, Spk{ki}(tmpi1));
            [ssi_j, ci_j, logC_kj] = removeSpike(Spk{kj},c(kj,:),logC(kj),efs_c{kj},a_c{kj}(tmpi2),taus_c{kj},traces_c(kj,:),Spk{kj}(tmpi2),tmpi2, Dt, A, Spk{kj}(tmpi2));

            %proposal for voltage
            v_ = v;
            [~, vi_i, diff_v_] = removeSpike_ar(Spk{ki},v(ki,:),diff_v,efs_v{ki},a_v(ki),taus_v{ki},trace_v,conversion_ratio*Spk{ki}(tmpi1),tmpi1, Dt, A, baseline_v + v_alt + sum(v)-v(ki,:));
            v_(ki,:) = vi_i;
            [~, vi_j, diff_v_] = removeSpike_ar(Spk{kj},v_(kj,:),diff_v_,efs_v{kj},a_v(kj),taus_v{kj},trace_v,conversion_ratio*Spk{kj}(tmpi2),tmpi2, Dt, A, baseline_v + v_alt + sum(v_)-v_(kj,:));
            v_(kj,:) = vi_j;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Add (tmpj1 to kj and tmpj2 to ki)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

            tmpj1 = Spk{ki}(tmpi1);
            tmpj2 = Spk{kj}(tmpi2);

            if (any(abs(tmpj1-ssi_j)<exclusion_bound) || N(kj) >= maxNevents) || (any(abs(tmpj2-ssi_i)<exclusion_bound) || N(ki) >= maxNevents) 
                continue
            else
                %proposal for calcium
                a_init1 = a_c{kj}(tmpi2);
                [ssi_j, ci_j, logC_kj] = addSpike(ssi_j,ci_j,logC_kj,efs_c{kj},a_init1,taus_c{kj},traces_c(kj,:),tmpj1,N(kj), Dt, A, tmpj1);

                a_init2 = a_c{ki}(tmpi1);
                [ssi_i, ci_i, logC_ki] = addSpike(ssi_i,ci_i,logC_ki,efs_c{ki},a_init2,taus_c{ki},traces_c(ki,:),tmpj2,N(ki), Dt, A, tmpj2);
                
                %proposal for voltage
                [~, vi_j, diff_v_] = addSpike_ar(ssi_j,v_(kj,:),diff_v_,efs_v{kj},a_v(kj),taus_v{kj},trace_v,conversion_ratio*tmpj1,N(kj), Dt, A, baseline_v + v_alt + sum(v_)-v_(kj,:)); 
                v_(kj,:) = vi_j;
                [~, vi_i, diff_v_] = addSpike_ar(ssi_i,v_(ki,:),diff_v_,efs_v{ki},a_v(ki),taus_v{ki},trace_v,conversion_ratio*tmpj2,N(ki), Dt, A, baseline_v + v_alt + sum(v_)-v_(ki,:)); 
                v_(ki,:) = vi_i;
            end
            
            %accept or reject     
            prior_ratio = 1;
            ratio_v = exp(sum((1/(2*vNoiseVar))*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) )));   
            ratio_c_i = exp(sum((1/(2*cNoiseVar(ki)))*(logC_ki-logC(ki))));
            ratio_c_j = exp(sum((1/(2*cNoiseVar(kj)))*(logC_kj-logC(kj))));
            ratio = ratio_c_i*ratio_c_j*ratio_v*prior_ratio;
                
            if (ratio>1)||(ratio>rand) %accept
                Spk{ki} = ssi_i;
                Spk{kj} = ssi_j;
                c(ki,:) = ci_i;
                c(kj,:) = ci_j;
                logC(ki) = logC_ki;
                logC(kj) = logC_kj;
                v = v_;
                diff_v = diff_v_;
                a_c{ki}(tmpi1) = [];
                a_c{kj}(tmpi2) = [];
                a_c{kj} = [a_c{kj} a_init1];
                a_c{ki} = [a_c{ki} a_init2];
                swapMoves = swapMoves + [1 1];
            else
                %reject - do nothing
                swapMoves = swapMoves + [0 1];
            end
            N(ki) = length(Spk{ki});
            N(kj) = length(Spk{kj});
            Nv = sum(N) + N_alt;
        end
        
        
        
        
        if useExtraChannel
        for ii = 1:K
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % swap from extra channel to a random channel
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        kj = randi(K,1);
        if ~isempty(sti)
            tmpi = randi(N_alt);%pick one of the spikes at random
       
            %try dropping tmpi from ki and adding to kj (accept or reject)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Drop (from alt)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %proposal for voltage
            [si_i, vi_i, diff_v_] = removeSpike_ar(sti,v_alt,diff_v,efs_alt{tmpi},ati(tmpi),taus_alt{tmpi},trace_v,sti(tmpi),tmpi, Dt, A, baseline_v + sum(v));

            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Add (to kj)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

            tmpj = sti(tmpi)/conversion_ratio;

            if (any(abs(tmpj-Spk{kj})<exclusion_bound) || N(kj) >= maxNevents)  
                continue
            else
                %proposal for calcium
                if isempty(Spk{kj})
                    a_init = max(traces_c(kj,max(1,floor(tmpj)))/A - baseline_c(kj) + a_c_std*randn,a_c_min);%propose an initial amplitude for it
                else
                    a_init = a_c{kj}(1);
                end
                [ssi_j, ci_j, logC_kj] = addSpike(Spk{kj},c(kj,:),logC(kj),efs_c{kj},a_init,taus_c{kj},traces_c(kj,:),tmpj,N(kj)+1, Dt, A, tmpj);

                %proposal for voltage
                [~, vi_j, diff_v_] = addSpike_ar(Spk{kj},v(kj,:),diff_v_,efs_v{kj},a_v(kj),taus_v{kj},trace_v,conversion_ratio*tmpj,N(kj)+1, Dt, A, baseline_v + vi_i + sum(v)-v(kj,:)); 
            end
            %accept or reject     
            prior_ratio = 1;
            ratio_v = exp(sum((1/(2*vNoiseVar))*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) )));   
            ratio_c_j = exp(sum((1/(2*cNoiseVar(kj)))*(logC_kj-logC(kj))));
            ratio = ratio_c_j*ratio_v*prior_ratio;
            if (ratio>1)||(ratio>rand) %accept
                sti = si_i;
                Spk{kj} = ssi_j;
                c(kj,:) = ci_j;
                logC(kj) = logC_kj;
                v_alt = vi_i;
                v(kj,:) = vi_j;
                diff_v = diff_v_;
                ati(tmpi) = [];
                a_c{kj} = [a_c{kj} a_init];
                swapMoves = swapMoves + [1 1];
            else
                %reject - do nothing
                swapMoves = swapMoves + [0 1];
            end
            N(kj) = length(Spk{kj});
            N_alt = length(sti);
            Nv = sum(N) + N_alt;
        end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % swap from random channel to extra channel
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for ki = 1:K %the guy to swap from
            if isempty(Spk{ki})
                continue
            else
                tmpi = randi(N(ki));%pick one of the spikes at random
            end

            %try dropping tmpi from ki and adding to kj (accept or reject)\

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Drop (from ki)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %proposal for calcium
            [ssi_i, ci_i, logC_ki] = removeSpike(Spk{ki},c(ki,:),logC(ki),efs_c{ki},a_c{ki}(tmpi),taus_c{ki},traces_c(ki,:),Spk{ki}(tmpi),tmpi, Dt, A, Spk{ki}(tmpi));

            %proposal for voltage
            [~, vi_i, diff_v_] = removeSpike_ar(Spk{ki},v(ki,:),diff_v,efs_v{ki},a_v(ki),taus_v{ki},trace_v,conversion_ratio*Spk{ki}(tmpi),tmpi, Dt, A, baseline_v + v_alt + sum(v)-v(ki,:));

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Add (to alt)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

            tmpj = Spk{ki}(tmpi);

            %proposal for voltage
            [sti_, vi_j, diff_v_] = addSpike_ar(sti,v_alt,diff_v_,efs_v{ki},a_v(ki),taus_v{ki},trace_v,conversion_ratio*tmpj,N_alt+1, Dt, A, baseline_v + sum(v) + vi_i - v(ki,:));
%             end

            %accept or reject  
            prior_ratio = 1;
            ratio_v = exp(sum((1/(2*vNoiseVar))*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) )));   
            ratio_c_i = exp(sum((1/(2*cNoiseVar(ki)))*(logC_ki-logC(ki))));
            ratio = ratio_c_i*ratio_v*prior_ratio;
            
            if (ratio>1)||(ratio>rand) %accept
                Spk{ki} = ssi_i;
                sti = sti_;
                c(ki,:) = ci_i;
                logC(ki) = logC_ki;
                v(ki,:) = vi_i;
                v_alt = vi_j;
                diff_v = diff_v_;
                a_c{ki}(tmpi) = [];
                ati = [ati a_v(ki)];
                taus_alt{N_alt+1} = taus_v{ki};
                efs_alt{N_alt+1} = efs_v{ki};                    
                swapMoves = swapMoves + [1 1];
            else
                %reject - do nothing
                swapMoves = swapMoves + [0 1];
            end
            N(ki) = length(Spk{ki});
            N_alt = length(sti);
            Nv = sum(N) + N_alt;
        end
        
        
        
        for ii = 1:K
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % bidirectional moves (random channel and extra)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            kj = ii;
            
            if isempty(sti) || isempty(Spk{kj})
                continue
            else
                tmpi1 = randi(N_alt);%pick one of the spikes at random
%                 tmpi2 = randi(N(kj));%pick one of the spikes at random
                [~, tmpi2] = min( abs(sti(tmpi1)/conversion_ratio-Spk{kj}) );% pick nearest spike
            end

            %try dropping tmpi from alt and adding to kj (accept or reject)\

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Drop (tmpi1 from alt, tmpi2 from kj)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %proposal for calcium
            [ssi_j, ci_j, logC_kj] = removeSpike(Spk{kj},c(kj,:),logC(kj),efs_c{kj},a_c{kj}(tmpi2),taus_c{kj},traces_c(kj,:),Spk{kj}(tmpi2),tmpi2, Dt, A, Spk{kj}(tmpi2));

            %proposal for voltage
            [sti_, v_alt_, diff_v_] = removeSpike_ar(sti,v_alt,diff_v,efs_alt{tmpi1},ati(tmpi1),taus_alt{tmpi1},trace_v,sti(tmpi1),tmpi1, Dt, A, baseline_v + sum(v));
            v_ = v;
            [~, vi_j, diff_v_] = removeSpike_ar(Spk{kj},v_(kj,:),diff_v_,efs_v{kj},a_v(kj),taus_v{kj},trace_v,conversion_ratio*Spk{kj}(tmpi2),tmpi2, Dt, A, baseline_v + v_alt_ + sum(v_)-v_(kj,:));
            v_(kj,:) = vi_j;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Add (tmpj1 to kj and tmpj2 to ki)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

            tmpj1 = sti(tmpi1)/conversion_ratio;
            tmpj2 = Spk{kj}(tmpi2);

            if (any(abs(tmpj1-ssi_j)<exclusion_bound) || N(kj) >= maxNevents) || (any(abs(tmpj2-sti_/conversion_ratio)<exclusion_bound) || N_alt >= maxNevents) 
                continue
            else
                %proposal for calcium
                a_init1 = a_c{kj}(tmpi2);
                [ssi_j, ci_j, logC_kj] = addSpike(ssi_j,ci_j,logC_kj,efs_c{kj},a_init1,taus_c{kj},traces_c(kj,:),tmpj1,N(kj), Dt, A, tmpj1);

                %proposal for voltage
                [~, vi_j, diff_v_] = addSpike_ar(ssi_j,v_(kj,:),diff_v_,efs_v{kj},a_v(kj),taus_v{kj},trace_v,conversion_ratio*tmpj1,N(kj), Dt, A, baseline_v + v_alt_ + sum(v_)-v_(kj,:)); 
                v_(kj,:) = vi_j;
                [sti_, v_alt_, diff_v_] = addSpike_ar(sti_,v_alt_,diff_v_,efs_v{kj},a_v(kj),taus_v{kj},trace_v,conversion_ratio*tmpj2,N_alt, Dt, A, baseline_v + sum(v_)); 
            end
            
            %accept or reject     
            prior_ratio = 1;
            ratio_v = exp(sum((1/(2*vNoiseVar))*( predAR(diff_v_,phi,p,1) - predAR(diff_v,phi,p,1) )));   
            ratio_c_j = exp(sum((1/(2*cNoiseVar(kj)))*(logC_kj-logC(kj))));
            ratio = ratio_c_j*ratio_v*prior_ratio;
            if (ratio>1)||(ratio>rand) %accept
                sti = sti_;
                Spk{kj} = ssi_j;
                c(kj,:) = ci_j;
                logC(kj) = logC_kj;
                v_alt = v_alt_;
                v(kj,:) = vi_j;
                diff_v = diff_v_;
                ati(tmpi1) = [];
                ati = [ati a_v(kj)];
                a_c{kj}(tmpi2) = [];
                a_c{kj} = [a_c{kj} a_init1];
                display('swapped')
%                 swapMoves = swapMoves + [1 1];
            else
                %reject - do nothing
%                 swapMoves = swapMoves + [0 1];
            end
            N(kj) = length(Spk{kj});
            N_alt = length(sti);
            Nv = sum(N) + N_alt;
        end
                
                
        end
    end
    
        
        

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    
       
    % delays and delay updates
    
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % re-estimate the noise model - ephys
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % re-estimate the process parameters
    %%%%%%%%%%%%%%%%
    % estimate phi (ignore initial condition boundary effects)
    %%%%%%%%%%%%%%%%
    if p>0 %&& i>(nsweeps/100)
        e = diff_v'; % this is Tx1 (after transpose)
        E = [];
        for ip = 1:p
            E = [E e((p+1-ip):(end-ip))];
        end
        e = e((p+1):end);

        Phi_n = Phi_0 + vNoiseVar^(-1)*(E'*E); %typo in ref '94 paper

        phi_cond_mean = Phi_n\(Phi_0*phi_0 + vNoiseVar^(-1)*E'*e);

        sample_phi = 1;
        while sample_phi
            phi = [1 mvnrnd(phi_cond_mean,inv(Phi_n))];

            phi_poly = -phi;
            phi_poly(1) = 1;
            if all(abs(roots(phi_poly))<1) %check stability roots of z^p - phi_1 z^{p-1} - phi_2 z^{p-1}...
                sample_phi = 0;
            end
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % estimate noise - ephys
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % re-estimate the noise variance
    if i>1
        df = (numel(trace_v)); %DOF (possibly numel(ci(ti,:))-1)
        d1 = -predAR(diff_v,phi,p,1)/df; 
        nu0 = nu_0; %nu_0 or 0
        d0 = sig2_0; %sig2_0 or 0
        
        A_samp = 0.5 * (df + nu0); %nu0 is prior
        B_samp = 1/(0.5 * df * (d1 + d0)); %d0 is prior
        vNoiseVar = 1/gamrnd(A_samp,B_samp); %this could be inf but it shouldn't be
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % estimate noise - calcium
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % re-estimate the noise variance
    if i>1
        for ki = 1:K
            df = (numel(c(ki,:))); %DOF (possibly numel(ci(ti,:))-1)
            d1 = sum((c(ki,:)-traces_c(ki,:)).^2)/df; %ML - DOF, init, baseline and each event amplitude
            nu0 = nu_0; %nu_0 or 0
            d0 = sig2_0; %sig2_0 or 0
            
            A_samp = 0.5 * (df + nu0); %nu0 is prior
            B_samp = 1/(0.5 * df * (d1 + d0)); %d0 is prior
            cNoiseVar(ki) = 1/gamrnd(A_samp,B_samp); %this could be inf but it shouldn't be
        end
    end
    
    

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

    %store things
    N_sto = [N_sto N];

    samples_a_v{i}  = a_v;
    samples_a_c{i}  = a_c;
    samples_b_v{i} = baseline_v;
    samples_b_c{i} = baseline_c;
    samples_s{i} = Spk;
    samples_v{i} = v;
    samples_v_alt{i} = v_alt;
    samples_c{i} = c;
    samples_tau_v{i} = taus_v;
    samples_tau_c{i} = taus_c;
    samples_phi{i} = phi;
    samples_c_noise{i} = cNoiseVar;
    samples_v_noise{i} = vNoiseVar;
    samples_s_alt{i} = sti;
    samples_a_alt{i} = ati;
    samples_tau_alt{i} = taus_alt;

    objective_v = [objective_v -predAR(diff_v,phi,p,1)];
    objective_c = [objective_c logC];

    figure(10)
    subplot(221)
    plot(traces_c' + 3000*repmat(0:(K-1),Tc,1),'k')
    hold on
    plot(c' + 3000*repmat(0:(K-1),Tc,1))
    hold off
    set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);
    ylabel('calcium imaging traces (a.u.)')
    subplot(223)
    plot(linspace(1,Tc,length(v)),trace_v,'k')
    hold on
    plot(linspace(1,Tc,length(v)),sum(v) + v_alt + baseline_v,'r')
    hold off
    set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);
    ylabel('electrophysiology (a.u.)')
    subplot(222)
    plot(linspace(1,Tc,length(v)),v_alt)
    if i==1
        set(gcf,'position',[1760         213        1388         557]);
    end
    drawnow
    
    
    if sum(ismember(indreporti,i))
        fprintf([num2str(indreport(ismember(indreporti,i)),2),', '])
    end
end

%% Vigi's Clean up
%details about what the mcmc did
%addMoves, dropMoves, and timeMoves give acceptance probabilities for each subclass of move
mcmc.addMoves=addMoves;
mcmc.timeMoves=timeMoves;
mcmc.dropMoves=dropMoves;
mcmc.ampMoves=ampMoves;
mcmc.tauMoves=tauMoves;
mcmc.swapMoves=swapMoves;
mcmc.N_sto=N_sto;%number of events

trials.amp_v=samples_a_v;
trials.base_v=samples_b_v;
trials.amp_c=samples_a_c;
trials.base_c=samples_b_c;
trials.tau_v=samples_tau_v;
trials.tau_c=samples_tau_c;
trials.phi=samples_phi;
trials.noise_c = samples_c_noise;
trials.noise_v = samples_v_noise;
trials.obj_c = objective_c;
trials.obj_v = objective_v;
trials.times = samples_s;
trials.sti = samples_s_alt;
trials.ati = samples_a_alt;
trials.tau_alt = samples_tau_alt;
trials.v = samples_v;
trials.c = samples_c;
trials.v_alt = samples_v_alt;





