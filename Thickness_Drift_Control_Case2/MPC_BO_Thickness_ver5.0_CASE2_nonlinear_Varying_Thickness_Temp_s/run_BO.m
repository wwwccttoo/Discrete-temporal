clc; clear all; close all;
%% Input Variables

load Res_STVBOwPOS64_nonlinear_2.mat
data_results = Res_STVBOwPOS64_nonlinear_2;

%% RT_controller
% run_no = 1;
% run_no = 40;
run_no = 50;
% run_no = 60;
% run_no = 70;
% run_no = 110;

Tref = data_results(run_no,2);
Iref = data_results(run_no,3);
Aref = [data_results(run_no,4) data_results(run_no,5) data_results(run_no,6)]';
dA = data_results(run_no,8);

Run_output = RT_control(Tref, Iref, Aref, dA);


%% Output Variables
% BO_output = Run_output.BO;


%% Plotting (Don't need to consider in your B.O. code)

scrsz = get(0,'ScreenSize');
font_size = 10;     % font size for plotting
line_width = 1.5;     % line width for plotting

sim_data  = Run_output.sim_data;
prob_info = Run_output.prob_info;

ctime = sim_data.ctime;
fprintf('Total Runtime: %.2f s\n', sum(ctime))
fprintf('Average Runtime: %.4f s\n', mean(ctime))

sim_ref = sim_data.Yrefsim + prob_info.xss(1:2); % reference was saved as deviation variable using the PREDICTION model as basis
Tplot = sim_data.Ysim(1,1:end-1) + prob_info.xssp(1);
I706plot = sim_data.Ysim(2,1:end-1) + prob_info.xssp(2);
I777plot = sim_data.Ysim(3,1:end-1) + prob_info.xssp(3);

t = linspace(0,length(sim_ref),length(sim_ref))*prob_info.ts;

dose_sim = sim_data.dose_sim(:,1:end-1);
DOSEplot = dose_sim;

Up = sim_data.Usim(1,1:end) + prob_info.ussp(1);
Uq = sim_data.Usim(2,1:end) + prob_info.ussp(2)+1.2;


%%
figure0 = figure('Position', [2 scrsz(1)/2 scrsz(3)/4 scrsz(4)/2]);
ax1 = subplot(5,1,1);
plot(t, DOSEplot, 'b-', 'LineWidth', line_width, 'DisplayName', 'DOSE')
% xlabel('Time (s)')
ylabel('Thickness(nm)'); grid;
legend()
set(gca, 'Fontsize', font_size)
ylim([0 1]);
ax2 = subplot(5,1,2);
plot(t, sim_ref(1,:), 'k--', 'LineWidth', line_width, 'DisplayName', 'Ref.')
hold on;
T_max = (prob_info.x_max(1)+prob_info.xss(1))*ones(size(Tplot));
% plot(t, T_max, 'r--', 'LineWidth', line_width, 'DisplayName', 'Constraint')
plot(t, Tplot, 'b-', 'LineWidth', line_width, 'DisplayName', 'T_{surf.}')
% xlabel('Time (s)')
ylabel('Temp.(℃)'); grid;
legend()
set(gca, 'Fontsize', font_size)
ylim([45 65]);
ax3 = subplot(5,1,3);
plot(t, sim_ref(2,:), 'k--', 'LineWidth', line_width, 'DisplayName', 'Ref.')
hold on;
plot(t, I706plot, 'b-', 'LineWidth', line_width, 'DisplayName', 'I_{He}')
% xlabel('Time (s)')
ylabel('Intensity(a.u)'); grid;
legend()
set(gca, 'Fontsize', font_size)
% ylim([1450 2000]);
ylim([1250 3000]);
ax4 = subplot(5,1,4);
% plot(t, sim_ref(2,:), 'k--', 'LineWidth', line_width, 'DisplayName', 'Ref.')
% hold on;
plot(t, I777plot, 'b-', 'LineWidth', line_width, 'DisplayName', 'I_{O2}')
% xlabel('Time (s)')
ylabel('Intensity(a.u)'); grid;
legend()
set(gca, 'Fontsize', font_size)
ylim([2000 5000]);
ax5 = subplot(5,1,5);
plot(t, Up, 'c-', 'LineWidth', line_width, 'DisplayName', 'Power')
hold on;
plot(t, Uq, 'r-', 'LineWidth', line_width, 'DisplayName', 'Flow-rate')
xlabel('Time (s)')
ylabel('Power[W],\newline Flow-rate(SLM)'); grid;
legend()
set(gca, 'Fontsize', font_size)
ylim([0 8]);
linkaxes([ax1 ax2 ax3 ax4 ax5],'x')
xlim([0 100])
