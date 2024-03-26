function plotInput(c, colors, lineWidth)

figure; hold on;
for i = 1:length(c)
    ns = c{i}.nu;
    time = 0:c{i}.sim.Nsim;
    for n = 1:c{i}.sim.Nrep
        for is = 1:ns
            subplot(ns,1,is); hold on;
            stairs(time,[c{i}.data.U(is,:,n), c{i}.data.U(is,end,n)],colors{i},'linewidth',lineWidth{i})
            plot([0, time(end)], [c{1}.bounds.u_min(is), c{1}.bounds.u_min(is)], '--k', 'linewidth', 2)
            plot([0, time(end)], [c{1}.bounds.u_max(is), c{1}.bounds.u_max(is)], '--k', 'linewidth', 2)
        end
    end
end
for is = 1:ns
    subplot(ns,1,is); hold on;
    set(gcf,'color','w');
    set(gca,'FontSize',20)
    xlabel('time')
    ylabel(['u' num2str(is)])
end

end
