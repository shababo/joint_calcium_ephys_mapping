function [newCalcium, newLL] = remove_init(oldCalcium,oldLL,filter,amp,tau,obsCalcium,Dt,A)
    
    timeToRemove = 0;
    
    tau_d = tau(2);
    
    ef_d = filter{2};
    
    %use infinite precision to scale the precomputed FIR approximation to the calcium transient    
    wk_d = amp*A*exp((timeToRemove - Dt*ceil(timeToRemove/Dt))/tau_d);
    
    %%%%%%%%%%%%%%%%%
    %handle ef_d next
    newCalcium = oldCalcium;
    tmp = 1+ (floor(timeToRemove):min((length(ef_d)+floor(timeToRemove)-1),length(newCalcium)-1));
    newCalcium(tmp) = newCalcium(tmp) - wk_d*ef_d(1:length(tmp));

%     %if you really want to, ef*ef' could be precomputed and passed in
%     relevantResidual = obsCalcium(tmp)-oldCalcium(tmp);
%     newLL = oldLL - ( wk_d^2*norm(ef_d(1:length(tmp)))^2 + 2*relevantResidual*(wk_d*ef_d(1:length(tmp))'));
%     %%%%%%%%%%%%%%%%%
    
    newLL = -sum((newCalcium-obsCalcium).^2);