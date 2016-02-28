function [newSpikeTrain, newTrace, newLL] = removeSpike(oldSpikeTrain,oldTrace,oldLL,filter,amp,tau,obsTrace,timeToRemove,indx,Dt,A,timeToRemove_warped)
    
    tau_h = tau(1);
    tau_d = tau(2);
    
    ef_h = filter{1};
    ef_d = filter{2};
    
    newSpikeTrain = oldSpikeTrain;
    newSpikeTrain(indx) = [];
    
    %use infinite precision to scale the precomputed FIR approximation to the calcium transient    
    wk_h = amp*A*exp((timeToRemove_warped - Dt*ceil(timeToRemove_warped/Dt))/tau_h);
    wk_d = amp*A*exp((timeToRemove_warped - Dt*ceil(timeToRemove_warped/Dt))/tau_d);
    
    
    %%%%%%%%%%%%%%%%%
    %handle ef_h first
    newTrace = oldTrace;
    tmp = 1+ (floor(timeToRemove_warped):min((length(ef_h)+floor(timeToRemove_warped)-1),length(newTrace)-1));
    newTrace(tmp) = newTrace(tmp) - wk_h*ef_h(1:length(tmp));

    relevantResidual = obsTrace(tmp)-oldTrace(tmp);
    newLL = oldLL - ( wk_h^2*norm(ef_h(1:length(tmp)))^2 + 2*relevantResidual*(wk_h*ef_h(1:length(tmp))'));
    oldLL = newLL;
    %%%%%%%%%%%%%%%%%
    
    oldTrace = newTrace;            
    
    %%%%%%%%%%%%%%%%%
    %handle ef_d next
    tmp = 1+ (floor(timeToRemove_warped):min((length(ef_d)+floor(timeToRemove_warped)-1),length(newTrace)-1));
    newTrace(tmp) = newTrace(tmp) - wk_d*ef_d(1:length(tmp));

    relevantResidual = obsTrace(tmp)-oldTrace(tmp);
    newLL = oldLL - ( wk_d^2*norm(ef_d(1:length(tmp)))^2 + 2*relevantResidual*(wk_d*ef_d(1:length(tmp))'));
    %%%%%%%%%%%%%%%%
    

    
