function [Trs,tau,offsets,rescalings,period,refT]=reshapeData(time,traces,tau)
%format multiple trial data
tinit=time(:,1);
[~,inds]=sort(tinit,'descend');
traces=traces(inds,:);
time=time(inds,:);
refT=time(1,:);
% trial 1 is a reference -- events can't be outside the time length of this trial
period = diff(time(:,1:2),1,2)';%frame period. only look at the first ones, will be the same throughout
rescalings=period(1)./period;
offsets = ((time(:,1)-time(1,1))/period(1))';%the units are percentage of a bin size
%convert taus
tau = tau/mean(period);
%convert data to a cell array and remove NaNs
Trs=cellfun(@(x)x(~isnan(x)),num2cell(traces,2),'UniformOutput',0);
