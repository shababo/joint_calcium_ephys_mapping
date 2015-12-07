function [CaF,tau,offsets,rescalings,period,refT]=reshapeData(time,traces,tau)

tinit=time(:,1);
[~,inds]=sort(tinit,'descend');
traces=traces(inds,:);
time=time(inds,:);
refT=time(1,:);
% trial 1 is a reference -- bursts can't be outside the time length of this trial
period = diff(time(:,1:2),1,2)';%frame period. only look at the first ones, will be the same throughout
% [I,J]=find(~isnan(time));
% [~,m] = unique(I, 'last');
% lInd=sub2ind(size(time),1:size(time,1),J(m)');
% totalT=time(lInd)'-time(:,1);%how much time did the trial take
% relT=totalT/totalT(1);%reference that to our first trial
% rescalings = 1./relT;%once we find things in the ref frame of 1, apply to others
rescalings=period(1)./period;
offsets = ((time(:,1)-time(1,1))/period(1))';%the units are percentage of a bin size
%convert taus
tau = tau/mean(period);
%convert calcium data to a cell array and remove NaNs
CaF=cellfun(@(x)x(~isnan(x)),num2cell(traces,2),'UniformOutput',0);
