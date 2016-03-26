addpath('./Fcns/')


%% This code is intended to be run as cell blocks


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Simulate presynaptic data (as in Calcium imaging)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulate K calcium imaging observed neurons
K = 6;

% Simulation: Given sampling, what indicator timecourse. Also depends on spike rate
% Sampling rate
% Noise level
% Time constants
% Firing rate
% Poisson or periodic

T = 200; %bins - start not too long
binSize = 35; %ms
tau_r = 100; %ms
tau_f = 300; %ms
firing_rate = .45; %spike/sec 
c_noise = 180; %calcium signal std
baseline = 300;
A = 1200; %magnitude scale of spike response

% key variables
Y = zeros(K,T);
C = zeros(K,T);
Spk = cell(1,K);

periodic = 0; %if zero, uses poisson spiketrain. (For periodic, we do not exploit this information in inference, it is just to make simple simulated data).

tau = [tau_r/binSize tau_f/binSize]; %time constants in bin-units

% compute exponential filter(s)
ef=genEfilt(tau,T);

n_spike = firing_rate*(T*binSize*1e-3);
p_spike = n_spike/T;

times = cumsum(binSize*1e-3*ones(1,T),2); % in sec

a_min = .3;
a_max = 3.5;
nc = 1; %trials

for ki = 1:K
    ssi = [];
    % startingSpikeTimes = [10 20];
    startingSpikeTimes = [];
    if periodic
        s_int = T/n_spike;
        startingSpikeTimes = [startingSpikeTimes s_int:s_int:(T-10)];
    else
%         startingSpikeTimes = times(rand(1,T)<p_spike)/(binSize*1e-3); % select bins
        startingSpikeTimes = T*rand(1,poissrnd(n_spike));
    end
    ci = baseline*ones(nc,T); %initial calcium is set to baseline 

    offsets = zeros(nc,1);
    rescalings = ones(nc,1);

    st_std = 0; %across trials
    ati = cell(nc,1); % array of lists of individual trial spike times
    ati_ = cell(nc,1); % array of lists of individual trial spike times
    sti = cell(nc,1); % array of lists of individual trial spike times
    sti_ = cell(nc,1); % array of lists of individual trial spike times

    N = 0;
    Dt = 1;

    for i = 1:length(startingSpikeTimes)        
        tmpi = startingSpikeTimes(i); 
        ssi_ = [ssi tmpi];
        cti_ = ci;
        a_init = 1;%a_min + (a_max-a_min)*rand;
        ati_ = ati;
        logC_ = 0;
        for ti = 1:nc
            tmpi_ = tmpi+(st_std*randn);
            [si_, ci_, logC_] = addSpike(sti{ti},ci(ti,:),logC_,ef,a_init,tau,ci(ti,:),tmpi_, N+1, Dt, A, (tmpi_-offsets(ti))*rescalings(ti)); %adds all trials' spikes at same time
            sti_{ti} = si_;
            cti_(ti,:) = ci_;
            ati_{ti} = [ati_{ti} a_init];
        end
        ati = ati_;
        ssi = ssi_;
        sti = sti_;
        ci = cti_;
        logC = logC_;
        N = length(ssi); %number of spikes in spiketrain
    end

    y = ci + c_noise*randn(nc,T);
    
    Y(ki,:) = y;
    C(ki,:) = ci;
    Spk{ki} = startingSpikeTimes;
end

figure;
subplot(211)
plot(C' + 3000*repmat(0:(K-1),T,1))
subplot(212)
plot(Y' + 3000*repmat(0:(K-1),T,1))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Simulate (postsynaptic) patched cell
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% V_t = [sum_ij a_ij k_j(t-t_ij-d_j)] + sum_i a_i k_i(t-t_i) + noise.

% observation params
% binSize_post = .1; %ms %original
binSize_post = .05; %ms %same as Ben's stuff
T_post = T*binSize/binSize_post; 
tau_r_post = .5; %ms
tau_f_post = 1; %ms
baseline_post = 20;
A_post = 4; %magnitude scale of spike response
Dt = 1;

V = baseline_post*ones(K,T_post);

% noise process params
p = 2;
phi = [1, .3, .35]; %reasonable example
v_noise = 2; 
U = v_noise*randn(T_post,1);
er = zeros(T_post,1);

%AR(2) noise
for t = (1+p):(T_post+p)
    er(t) = phi*[U(t-p); er(t-1:-1:(t-p))];
end

% filter(s) for simulated responses
tau_post = [tau_r_post/binSize_post tau_f_post/binSize_post]; %time constants in bin-units

fBins = tau_f_post*70;
ef = genEfilt_ar(tau_post,fBins);%exponential filter
ef_init = ef;
efs = cell(1,K);

%loop over neurons and spikes per neuron (also have an extra element in the
%outer loop for random non-calcium-observed neurons.)
for i = 1:K
    tau_post_i = [(tau_r_post+.5*rand)/binSize_post (tau_f_post+2*rand)/binSize_post]; %time constants in bin-units
    fBins_i = tau_post_i*70;
    ef_i = genEfilt_ar(tau_post,fBins_i);%exponential filter
    efs{i} = ef_i;
    nispks = Spk{i}*binSize/binSize_post;
    a_init = 1 + 3*rand; %was deterministic, for previous (now varies across neurons)
    for j = 1:length(nispks)
        tmpi = nispks(j); 
        sti_ = [sti tmpi];
        %must add spike to each trial (at mean location or sampled -- more appropriate if sampled)
        [~, V(i,:), ~] = addSpike_ar([],V(i,:),0,efs{i},a_init,tau_post,V(i,:),tmpi, 1, Dt, A_post); %adds all trials' spikes at same time
    end
end

%
%load simFig_example.mat mc
V_o = sum(V) + er((1+p):end,:)';
figure
subplot(311)
% plot(V' + A_post*2*repmat(0:(K-1),T_post,1))
plot(sum(V))
subplot(312)
plot(er((1+p):end))
subplot(313)
plot(V_o)


%% Initialize by running separate calcium imaging inference ? -- omitted for now
% there are calcium only params -- noise, time constants of each neuron's response
% there are ephys only params -- noise, time constants of each neuron's response, delays (omitted)
% there are joint parameters -- spike times, connection strength

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Inference based on postsynaptic (ephys data only) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% skip if interested in joint inference only

params.p_spike = 5.0000e-04/2;
params.tau_min = 5; %bins
params.tau_max = 100; %bins
params.dt = binSize*1e-3; % in terms of calcium part
params.p = 2;
params.a_min = 2;
params.eb = 0; % exclusion bound between spikes (in units of calcium bins)
Tguess = [];

trace = V_o;

subset = 1:3e4; % in order to go faster, use only subset of the trace

[trials, mcmc, params]  = sampleParams_ARnoise(trace(subset), tau_post, Tguess, params);

%% visualize (last sample and true visualized here for simplicity)
%...note last sample doesn't always capture posterior events accurately
% NOTE: it is useful to view this figure as a wide and short window



% PlotMCMC_ar %requires storing curves (commented out in
% sampleParams_ARnoise...only okay for small example)

convBinPre2s = binSize*1e-3;
convBinPost2s = binSize_post*1e-3;

true_spikes = cell2mat(Spk)*convBinPre2s; %in seconds
true_spikes = true_spikes(true_spikes<max(subset)*convBinPost2s);
figure;plot(convBinPost2s*subset,trace(subset),'k')
hold on
plot(convBinPost2s*trials.times{end},max(trace(1:3e4))*ones(1,length(trials.times{end})),'r.','markersize',10)
plot(true_spikes,max(trace(1:3e4))*ones(1,length(true_spikes)),'ro','markersize',10)
hold off
set(gcf,'position',[1848, 452, 1071, 230])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% inference w/r/t joint likelihood (using presynaptic and postsynaptic data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% inputs: V_o - voltage... Y - calcium... initial vals of taus, initial
% vals of times, other params

% want: times, time constants, baselines, amplitudes, noise
params.p_spike = 5.0000e-04;
params.tau_v_min = 5; %bins
params.tau_v_max = 100; %bins
params.tau_c_min = 2;
params.tau_c_max = 10;
params.dt = binSize*1e-3; % in terms of calcium part
params.p = 2;
params.a_c_min = 800;
params.a_v_min = 2;
params.eb = 0; % exclusion bound between spikes (in units of calcium bins)
conversion_ratio = binSize/binSize_post;
Tguess = [];
tau_c = tau;
tau_v = tau_post;

%% sampleParams_joint3 has linked calcium amplitudes per channel

subtraces = 3:6;
[trials, mcmc, params]  = sampleParams_joint3(Y(subtraces,:),V_o,tau_c, 1.5*tau_v, Tguess, conversion_ratio, params);

%% Save if you like
% save example_realistic_test -v7.3 %without flag trials isn't saved

%% produce a figure
convBin2s = binSize*1e-3;
subtraces = 3:6;
% subtraces = 1:6;
figure(10)
% subplot(221)
subplot(211)
v_offset = 3000;
plot(times,Y(subtraces,:)' + v_offset*repmat(0:(length(subtraces)-1),T,1),'k')
colorOrder = get(gca, 'ColorOrder');
hold on
for ki = 1:length(subtraces)
    plot(convBin2s*Spk{subtraces(ki)},ones(1,length(Spk{subtraces(ki)}))*v_offset*(ki-1)-v_offset/10,'k.','MarkerSize',15) % true events
    plot(convBin2s*trials.times{end}{ki},ones(1,length(trials.times{end}{ki}))*v_offset*(ki-1)-3*v_offset/10,'.','color',colorOrder(ki,:),'MarkerSize',15) % inferred events
end
plot(times,trials.c{end}' + v_offset*repmat(0:(length(subtraces)-1),T,1))
hold off
set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);
ylabel('calcium imaging traces (a.u.)','fontsize',16)
xlim([0 max(times)])
% subplot(223)
set(gca,'fontsize',16)
axis tight
yl = get(gca,'ylim');
ylim([yl(1)-.05*(diff(yl)) yl(2)])
box off
set(gca,'ytick',[])

subplot(212)
plot(linspace(0,max(times),length(trials.v{end})),-V_o,'k')
hold on
plot(linspace(0,max(times),length(trials.v{end})),-(sum(trials.v{end}) + trials.v_alt{end} + trials.base_v{end}),'r')
for ki = 1:length(subtraces)
    plot(convBin2s*trials.times{end}{ki},ones(1,length(trials.times{end}{ki})) - trials.base_v{end} + 8 + 1*ki,'.','color',colorOrder(ki,:),'MarkerSize',15) % inferred events
end
plot(convBin2s*trials.sti{end}/conversion_ratio,ones(1,length(trials.sti{end})) - trials.base_v{end} + 8 ,'.','color', [0.5 0.5 0.5],'MarkerSize',15) % inferred events
hold off
set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);
ylabel('electrophysiology (a.u.)','fontsize',16)
xlabel('time (s)','fontsize',16)
xlim([0 max(times)])
% subplot(222)
% plot(trials.v_alt)
set(gcf,'position',[1760         213        1388         557]);
set(gca,'fontsize',16)
axis tight
yl = get(gca,'ylim');
ylim([yl(1)-.05*(diff(yl)) yl(2)+.05*(diff(yl))])
drawnow
box off
set(gca,'ytick',[])

% exportfig(gcf,['example_partial_obs_joint.eps'],'bounds','tight','Color','rgb','FontSize',1,'LockAxes',0);



