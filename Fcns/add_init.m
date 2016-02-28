function [newTrace, newLL] = add_init(oldTrace,oldLL,filter,amp,tau,obsTrace,Dt,A)

    timeToAdd = 0;
    
    tau_d = tau(2);
    
    ef_d = filter{2};
        
    %use infinite precision to scale the precomputed FIR approximation to the Trace transient
    wk_d = amp*A*exp((timeToAdd - Dt*ceil(timeToAdd/Dt))/tau_d);
    
    %%%%%%%%%%%%%%%%%
    newTrace = oldTrace;
    %%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%
    %handle ef_d 
    tmp = 1 + (floor(timeToAdd):min((length(ef_d)+floor(timeToAdd)-1),length(newTrace)-1));
    newTrace(tmp) = newTrace(tmp) + wk_d*ef_d(1:length(tmp));
    
    %%%%%%%%%%%%%%%%%
    
    
    newLL = -sum((newTrace-obsTrace).^2);
