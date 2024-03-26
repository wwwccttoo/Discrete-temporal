function plotDisturbanceEstimate(c, colors, lineWidth)

figure; hold on;
for i = 1:length(c)
    ns = c{i}.nd;
    time = 0:c{i}.sim.Nsim;
    for n = 1:c{i}.sim.Nrep
        for is = 1:ns
            subplot(ns,1,is); hold on;
            plot(time,c{i}.data.Dhat(is,:,n),colors{i},'linewidth',lineWidth{i})
        end
    end
end
for is = 1:ns
    subplot(ns,1,is); hold on;
    set(gcf,'color','w');
    set(gca,'FontSize',20)
    xlabel('time')
    ylabel(['dhat' num2str(is)])
end

end
