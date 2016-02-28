function [newTrace, newLL] = remove_base(oldTrace,oldLL,amp,obsTrace,A)
    
    newTrace = oldTrace;
    
    newTrace = newTrace - amp*A*ones(1,length(oldTrace));
   
    newLL = -sum((newTrace-obsTrace).^2);
