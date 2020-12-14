function [] = plot_results(out,out1,method1,out2,method2,out3,method3,out4,method4,out5,method5)
fig = figure(1);

data1 = (out1.f - out) / (out);
semilogy(1:length(data1), max(data1,0), '-.', 'color', [0.1 1 0.1], 'LineWidth',1);
hold on;
data2 = (out2.f - out) / (out);
semilogy(1:length(data2), max(data2,0), '-.', 'color', [0.1 0.1 1], 'LineWidth',1);
data3 = (out3.f - out) / (out);
semilogy(1:length(data3), max(data3,0), '-.', 'color', [0.6 0.1 0.6], 'LineWidth',1);
data4 = (out4.f - out) / (out);
semilogy(1:length(data4), max(data4,0), ':', 'color', [0.9 0.1 0.1], 'LineWidth',1);
data5 = (out5.f - out) / (out);
semilogy(1:length(data5), max(data5,0), ':', 'color', [0.2 0.8 0.2], 'LineWidth',1);

title('Convergence Process');
legend(method1,method2,method3,method4,method5);
ylabel('$(f(x^k) - f^*)/f^*$', 'fontsize', 13, 'interpreter', 'latex');
xlabel('Number of iterations');
hold off;

saveas(fig, '../result/convergence.png');
end