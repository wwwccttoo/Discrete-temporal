clc; clear all; close all;

%% System drift)

dA_init = 1.0;
dA_final = 1.5;
No_run_max  = 100;
iter_init = 10;



%% RT_controller

Results_final = [];

for k = 1:(No_run_max+iter_init)

    % if k < (No_run_default + 1)
    %     dA = 1.0;
    % elseif k > No_run_max + (No_run_default)
    %     dA = dA_final;
    % else
    %     dA = dA_init - (dA_init - dA_final)/(No_run_max-1)*(k-11);
    % end

    if k < (iter_init + 1)
        dA = dA_init;
    else
        dA = 1 - 0.2./(1.+exp(-0.15*(k-40-iter_init-1)));

    end
    dA_tot(k,1) = dA;

    % if k == 1
        Tref  = 50; % Desired surface temp.
        Iref  = 2000; % Desired intensity
        PTref = 93;   % Desired process time.
        Aref = [1.4016 -14.4395 12.5570]';
    % else
    %     Tref  = data_classic_BO(k-1,2);
    %     Iref  = data_classic_BO(k-1,3);
    %     PTref = data_classic_BO(k-1,4);
    % end

    % w_thick = 0.8;
    Run_output = RT_control_return(Tref, Iref, Aref, dA);
    Results_final(k).data = Run_output;
end

for k = 1:110
    yPrs(k) =  Results_final(k).data(1,1);
end

yPrs_avg = mean(yPrs)
yPrs_avg_diff = 1.0048- yPrs_avg
max(abs(yPrs - yPrs(1)))
std(yPrs)*3

figure
plot([1:110], yPrs); grid;

%% Output Variables

% for k = 1:length(Results_final)
%     MPC_output(k,1) = max(Results_final(k).data.sim_data.dose_sim(:,1:end-1));
% end



%% Plotting (Don't need to consider in your B.O. code)

scrsz = get(0,'ScreenSize');
font_size = 10;     % font size for plotting
line_width = 1.5;     % line width for plotting

iter_no = [1:length(yPrs)]';

%% MPC w/o B.O
figure0 = figure('Position', [2 scrsz(1)/2 scrsz(3)/4 scrsz(4)/2]);
subplot(5,1,1)
plot(iter_no, dA_tot, 'b-x', 'LineWidth', line_width, 'DisplayName', 'dA')
ylabel('dA[%]'); grid;
title('Uncertainty(drift)')
subplot(5,1,2)
plot(iter_no, yPrs, 'b-x', 'LineWidth', line_width, 'DisplayName', 'DOSE')
ylabel('Thickness[nm]'); grid;
title('Thickness')
ylim([0.8 1.3]);
% xlabel('Iteration No.')
subplot(5,1,3)
plot(iter_no, Tref*ones(length(iter_no),1), 'b-x', 'LineWidth', line_width)
ylabel('Temp.[℃]'); grid;
% xlabel('Iteration No.')
title('Surface Temperature')
ylim([35 65]);
subplot(5,1,4)
plot(iter_no, Iref*ones(length(iter_no),1), 'b-x', 'LineWidth', line_width)
ylabel('Intensity'); grid;
xlabel('Iteration No.')
title('He Intensity')
ylim([1500 3000]);
subplot(5,1,5)
plot(iter_no, PTref*ones(length(iter_no),1), 'b-x', 'LineWidth', line_width)
ylabel('Time[s]'); grid;
ylim([90 100])
xlabel('Iteration No.')
title('Process Time')


%%
% sim_data  = Run_output.sim_data;
% prob_info = Run_output.prob_info;
%
% ctime = sim_data.ctime;
% fprintf('Total Runtime: %.2f s\n', sum(ctime))
% fprintf('Average Runtime: %.4f s\n', mean(ctime))
%
% sim_ref = sim_data.Yrefsim + prob_info.xss(1:2); % reference was saved as deviation variable using the PREDICTION model as basis
% Tplot = sim_data.Ysim(1,1:end-1) + prob_info.xssp(1);
% I706plot = sim_data.Ysim(2,1:end-1) + prob_info.xssp(2);
% I777plot = sim_data.Ysim(3,1:end-1) + prob_info.xssp(3);
%
% t = linspace(0,length(sim_ref),length(sim_ref))*prob_info.ts;
%
% dose_sim = sim_data.dose_sim(:,1:end-1);
% DOSEplot = dose_sim;
%
% Up = sim_data.Usim(1,1:end) + prob_info.ussp(1);
% Uq = sim_data.Usim(2,1:end) + prob_info.ussp(2)+1.2;


%%
% figure0 = figure('Position', [2 scrsz(1)/2 scrsz(3)/4 scrsz(4)/2]);
% subplot(5,1,1)
% plot(t, DOSEplot, 'b-', 'LineWidth', line_width, 'DisplayName', 'DOSE')
% % xlabel('Time (s)')
% ylabel('Thickness(nm)'); grid;
% legend()
% set(gca, 'Fontsize', font_size)
% ylim([0 1]);
% subplot(5,1,2)
% plot(t, sim_ref(1,:), 'k--', 'LineWidth', line_width, 'DisplayName', 'Ref.')
% hold on;
% T_max = (prob_info.x_max(1)+prob_info.xss(1))*ones(size(Tplot));
% % plot(t, T_max, 'r--', 'LineWidth', line_width, 'DisplayName', 'Constraint')
% plot(t, Tplot, 'b-', 'LineWidth', line_width, 'DisplayName', 'Temp_{surf.}')
% % xlabel('Time (s)')
% ylabel('Temp.(℃)'); grid;
% legend()
% set(gca, 'Fontsize', font_size)
% ylim([45 65]);
% subplot(5,1,3)
% plot(t, sim_ref(2,:), 'k--', 'LineWidth', line_width, 'DisplayName', 'Ref.')
% hold on;
% plot(t, I706plot, 'b-', 'LineWidth', line_width, 'DisplayName', 'He_{I706}')
% % xlabel('Time (s)')
% ylabel('Intensity(a.u)'); grid;
% legend()
% set(gca, 'Fontsize', font_size)
% % ylim([1450 2000]);
% ylim([1250 2800]);
% subplot(5,1,4)
% % plot(t, sim_ref(2,:), 'k--', 'LineWidth', line_width, 'DisplayName', 'Ref.')
% % hold on;
% plot(t, I777plot, 'b-', 'LineWidth', line_width, 'DisplayName', 'O_{I777}')
% % xlabel('Time (s)')
% ylabel('Intensity(a.u)'); grid;
% legend()
% set(gca, 'Fontsize', font_size)
% ylim([2000 5000]);
% subplot(5,1,5)
% plot(t, Up, 'c-', 'LineWidth', line_width, 'DisplayName', 'Power')
% hold on;
% plot(t, Uq, 'r-', 'LineWidth', line_width, 'DisplayName', 'Flow-rate')
% xlabel('Time (s)')
% ylabel('Power[W],\newline Flow-rate(SLM)'); grid;
% legend()
% set(gca, 'Fontsize', font_size)
% ylim([0 8]);
