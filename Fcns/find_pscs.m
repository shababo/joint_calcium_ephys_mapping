function event_onsets = find_pscs(trace, dt, tau, amp_thesh, conv_thresh, low_passed, plot_figs)

t_vector = (0:length(trace)-1)*dt;

noise_sd = std(trace - smooth(trace,100,'sgolay',4)');

prev_samples = floor(3*tau/dt);
template = alpha_synapse(t_vector,0,tau,-1);
template = [zeros(1,prev_samples) template];

if low_passed
    template = template - smooth(template,500,'sgolay',2)';
end

template(floor(13*tau/dt):end) = [];

template = template *8;

colored_noise = conv(normrnd(0,noise_sd,size(trace)),template,'same');

convolution = conv([trace zeros(1,length(template))],template,'same');
if length(convolution) > length(trace)
    convolution(length(trace)+1:end) = [];
end

window_size = floor(2*tau/dt);
if mod(window_size,2) == 0
    window_size = window_size + 1;
end


thresh_vec = smooth([(trace - median(trace)) zeros(1,window_size)],window_size)';
thresh_vec(1:ceil(tau/dt)-1) = [];
if length(thresh_vec) > length(trace)
    thresh_vec(length(trace)+1:end) = [];
end

thresh_vec = thresh_vec < -amp_thesh;

convolution = smooth(convolution,floor(tau/dt))';

conv_thresh = conv_thresh*std(colored_noise);

event_onsets = find(convolution(2:end-1) > convolution(1:end-2) &...
                    convolution(2:end-1) > convolution(3:end)   &...
                    thresh_vec(2:end-1));
                
event_onsets = event_onsets + 1;


