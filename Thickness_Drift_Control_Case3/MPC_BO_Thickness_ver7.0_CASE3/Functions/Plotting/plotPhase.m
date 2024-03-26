function plotPhase(c, colors, lineWidth)

figure; hold on;
for i = 1:length(c)
    Nrep = c{i}.sim.Nrep;
    for n = 1:Nrep
        for is = 1:c{i}.nyc
            plot(c{i}.data.X(1,:,n),c{i}.data.X(2,:,n),colors{i},'linewidth',lineWidth{i})
        end
    end
end
plot([c{1}.bounds.x_min(1), c{1}.bounds.x_min(1)], [c{1}.bounds.x_min(2), c{1}.bounds.x_max(2)], '--k', 'linewidth', 2)
plot([c{1}.bounds.x_min(1), c{1}.bounds.x_max(1)], [c{1}.bounds.x_min(2), c{1}.bounds.x_min(2)], '--k', 'linewidth', 2)
plot([c{1}.bounds.x_min(1), c{1}.bounds.x_max(1)], [c{1}.bounds.x_max(2), c{1}.bounds.x_max(2)], '--k', 'linewidth', 2)
plot([c{1}.bounds.x_max(1), c{1}.bounds.x_max(1)], [c{1}.bounds.x_min(2), c{1}.bounds.x_max(2)], '--k', 'linewidth', 2)
set(gcf,'color','w');
set(gca,'FontSize',20)
xlabel('x1')
ylabel('x2')

end
