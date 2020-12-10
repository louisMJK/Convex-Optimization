function [] = plot_results(optval,out1,method1,out2,method2,out3,method3)
fig = figure(1);
data1 = (out1.f - optval) / optval;
semilogy(1:length(data1), max(data1,0), '-.', 'color', [0.9 0.3 0.3], 'LineWidth',1);
hold on;
data2 = (out2.f - optval) / optval;
semilogy(1:length(data2), max(data2,0), '-.', 'color', [0.1 0.1 0.9], 'LineWidth',1);
data3 = (out3.f - optval) / optval;
semilogy(1:length(data3), max(data3,0), '-.', 'color', [0.2 0.9 0.1], 'LineWidth',1);

title('Convergence process');
legend(method1,method2,method3);
ylabel('$(f(x^k) - f^*)/f^*$', 'fontsize', 13, 'interpreter', 'latex');
xlabel('Number of iterations');
hold off;

saveas(fig, strcat('../result/convergence.png'));
end