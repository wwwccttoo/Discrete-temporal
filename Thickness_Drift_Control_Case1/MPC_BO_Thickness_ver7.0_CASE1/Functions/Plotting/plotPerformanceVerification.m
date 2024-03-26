function plotPerformanceVerification(c, flags)

figure; hold on;
% plot performance verification controlled outputs
for i = 1:length(c)
    if flags{i}.performanceVerification == 1
        figure; hold on;
        Nrep = c{i}.mpc.perfVer.Nrep;
        ns = c{i}.nyc;
        time = 0:c{i}.mpc.perfVer.Nsim;
        for n = 1:Nrep
            for is = 1:c{i}.nyc
                subplot(ns,1,is); hold on;
                stairs(time,[c{i}.mpc.perfVer.data.Ytar(is,:,n),c{i}.mpc.perfVer.data.Ytar(is,end,n)],':k','linewidth',5)
                plot(time,c{i}.mpc.perfVer.data.Yc(is,:,n),'-b','linewidth',4)
                plot([0, time(end)], [c{1}.bounds.yc_min(is), c{1}.bounds.yc_min(is)], '--r', 'linewidth', 2)
                plot([0, time(end)], [c{1}.bounds.yc_max(is), c{1}.bounds.yc_max(is)], '--r', 'linewidth', 2)
            end
        end
        for is = 1:ns
            subplot(ns,1,is); hold on;
            set(gcf,'color','w');
            set(gca,'FontSize',20)
            xlabel('time')
            ylabel(['yc' num2str(is)])
        end
    end
end

end
