function [c, flags] = loadConfigurations(name)

warning('off')
load(name, 'c', 'flags')
warning('on')

for i = 1:length(c)
    % need to rebuild opti for tracker and mpc
    c{i} = getTargetTracker(c{i});
    c{i} = getMPC(c{i}, flags{i}.mpcType);
end

end
