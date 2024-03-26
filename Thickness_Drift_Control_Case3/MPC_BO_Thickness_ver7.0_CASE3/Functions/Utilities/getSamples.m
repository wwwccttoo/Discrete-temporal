function S = getSamples(s_min, s_max, N)

ns = length(s_min);
S = (s_max-s_min).*rand(ns,N) + s_min;

end
