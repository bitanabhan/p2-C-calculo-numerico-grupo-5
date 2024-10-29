clear all
close all
clc

columns_names = {'pelvic_incidence', 'pelvic_tilt', ...
                 'lumbar_lordosis_angle', 'sacral_slope', ...
                 'pelvic_radius', 'degree_spondylolisthesis', 'class'};

data = csvread('column_2C_cleaned.csv');

# removendo a linha com strings dos títulos
data(1, :) = [];

# início dos gráficos de dispersão dois a dois para verificação das relações
# esses gráficos são para vocês terem algum código inicial, não é obrigatório utiliza-los.
num_columns = length(columns_names)-2;

class_normal = data(data(:, end) == 0, 1:end);
class_anormal = data(data(:, end) == 1, 1:end);

for ii = 1:num_columns-1
    figure;
    scatter(class_normal(:, ii), class_normal(:, ii+1), 'b', 'x'); % Blue crosses for status 1
    hold on;
    scatter(class_anormal(:, ii), class_anormal(:, ii+1), 'r', 'o'); % Red circles for status 0
    xlabel(columns_names{ii});
    ylabel(columns_names{ii+1});
    title(['Scatter plot of ', columns_names{ii}, ' vs ', columns_names{ii+1}]);
    legend('Normal', 'Anormal');
    grid on
    hold off;
end

### início do código da prova (fique à vontade para comentar o código de plots dos scatters)


