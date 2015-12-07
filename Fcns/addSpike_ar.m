function [newSpikeTrain, newTrace, newLL] = addSpike_ar(oldSpikeTrain,oldTrace,oldLL,filter,amp,tau,obsTrace,timeToAdd,indx,Dt,A,baseline)

    if nargin<12
        baseline = zeros(size(oldTrace));
    end
    tau_h = tau(1);
    tau_d = tau(2);

    ef_h = filter{1};
    ef_d = filter{2};

    newSpikeTrain = [oldSpikeTrain(1:indx-1) timeToAdd oldSpikeTrain(indx:end)]; %adds in place rather than at end

    %use infinite precision to scale the precomputed FIR approximation to the calcium transient
    wk_h = amp*A*exp((timeToAdd - Dt*ceil(timeToAdd/Dt))/tau_h);
    wk_d = amp*A*exp((timeToAdd - Dt*ceil(timeToAdd/Dt))/tau_d);    
        
    %%%%%%%%%%%%%%%%%
    %handle ef_h first
    newTrace = oldTrace;
    tmp = 1 + (floor(timeToAdd):min((length(ef_h)+floor(timeToAdd)-1),length(newTrace)-1));
    newTrace(tmp) = newTrace(tmp) + wk_h*ef_h(1:length(tmp));

    newLL = oldLL;
    newLL(tmp) = obsTrace(tmp) - (newTrace(tmp) + baseline(tmp));

    oldTrace = newTrace;
    %%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%
    %handle ef_d next
    tmp = 1 + (floor(timeToAdd):min((length(ef_d)+floor(timeToAdd)-1),length(newTrace)-1));
    newTrace(tmp) = newTrace(tmp) + wk_d*ef_d(1:length(tmp));

    newLL(tmp) = obsTrace(tmp) - (newTrace(tmp) + baseline(tmp));
    





