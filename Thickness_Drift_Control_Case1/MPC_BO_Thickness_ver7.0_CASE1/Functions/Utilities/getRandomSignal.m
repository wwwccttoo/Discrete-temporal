function s_now = getRandomSignal(s_min, s_max, t_step, Nchange, Ntotal)

% get size of signal
ns = length(s_min);

% number of changes over simulation
n_change = ceil(Ntotal / Nchange);

% initialize list
s_list = zeros(n_change, ns);

% loop over number of elements
for i = 1:ns
    s_list(:,i) = (s_max(i)-s_min(i))*rand(n_change,1) + s_min(i);
end

% repeat last reference for plotting purposes
s_list = [s_list ; s_list(end,:)];

% define function that reads list
s_now = @(t)(s_list(floor(t/t_step/Nchange)+1,:)');

end
