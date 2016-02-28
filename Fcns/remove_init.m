function [newTrace, newLL] = remove_init(oldTrace,oldLL,filter,amp,tau,obsTrace,Dt,A)
    
    timeToRemove = 0;
    
    tau_d = tau(2);
    
    ef_d = filter{2};
    
    %use infinite precision to scale the precomputed FIR approximation to the Trace transient    
    wk_d = amp*A*exp((timeToRemove - Dt*ceil(timeToRemove/Dt))/tau_d);
    
    %%%%%%%%%%%%%%%%%
    %handle ef_d
    newTrace = oldTrace;
    tmp = 1+ (floor(timeToRemove):min((length(ef_d)+floor(timeToRemove)-1),length(newTrace)-1));
    newTrace(tmp) = newTrace(tmp) - wk_d*ef_d(1:length(tmp));

    %%%%%%%%%%%%%%%%%
    
    newLL = -sum((newTrace-obsTrace).^2);
