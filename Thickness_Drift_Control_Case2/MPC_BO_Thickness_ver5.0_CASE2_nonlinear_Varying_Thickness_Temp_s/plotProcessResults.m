function plotProcessResults(c, colors, lineWidth)

figure; hold on;
for i = 1:length(c)
    Nrep = c{i}.sim.Nrep;
    % ns = c{i}.nyc;
    time = 0:c{i}.sim.Nsim;
    for n = 1:Nrep
        % for is = 1:c{i}.nyc
            % subplot(ns,1,is); hold on;
            % stairs(time,[c{i}.data.Ytar(is,:,n),c{i}.data.Ytar(is,end,n)]+c{i}.xss(is),':k','linewidth',3)
            plot(time,[c{i}.data.Yprs(1,:,n), c{i}.data.Yprs(1,end,n)],colors{i},'linewidth',lineWidth{i})
            %plot([0, time(end)], [c{1}.bounds.yc_min(is), c{1}.bounds.yc_min(is)]+c{i}.xss(is), '--k', 'linewidth', 2)
            %plot([0, time(end)], [c{1}.bounds.yc_max(is), c{1}.bounds.yc_max(is)]+c{i}.xss(is), '--k', 'linewidth', 2)
        % end
    end
end
% for is = 1:ns
    % subplot(ns,1,is);
    hold on;
    set(gcf,'color','w');
    set(gca,'FontSize',20)
    xlabel('time')
    ylabel(['Thickness(nm)'])
% end

end
