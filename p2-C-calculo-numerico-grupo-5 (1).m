clear all
close all
clc

 ######### ANÁLISE 1 #########
#### COMEÇO 1.1 #####

% Regressão Linear
function [a, b] = linear_regression(x, y)
    n = length(x); % n é o número de elementos no vetor x (e também em y, pois têm o mesmo tamanho)

    % Calcula o coeficiente angular (b), que é a inclinação da reta de regressão
    b = (n * sum(x .* y) - sum(x) * sum(y)) / (n * sum(x.^2) - sum(x)^2);

    % Calcula o intercepto (a), que é o ponto onde a reta cruza o eixo y
    a = mean(y) - b * mean(x);
end

% Função para calcular o coeficiente de determinação (R²)
function r = calculate_pearson(x, y)
    r = corr(x, y)^2; % R² é o quadrado do coeficiente de correlação de Pearson
end


function coeffs = regressao_polinomial(x, y, grau)
    % Número de pontos de dados (tamanho do vetor x)
    m = length(x);

    % Grau do polinômio (definido pelo usuário)
    n = grau;

    % Constrói a matriz de Vandermonde, que será usada para calcular os
    % coeficientes do polinômio. A matriz de Vandermonde é essencial para
    % problemas de ajuste polinomial.
    V = zeros(m, n+1); % Inicializa a matriz com zeros (m linhas e n+1 colunas)

    % Preenche a matriz de Vandermonde com as potências de x
    for i = 0:n
        V(:, n+1 - i) = x.^i; % Cada coluna corresponde a uma potência de x
    end

    % Resolve as equações normais: A * coeffs = b, onde A = V' * V e b = V' * y
    A = V' * V; % Produto da transposta de V com V (gera o sistema de equações normais)
    b = V' * y(:); % Produto da transposta de V com y (coloca y como um vetor coluna)



    % Decomposição LU e solução usando funções personalizadas
   [L, U] = lu_decomposition(A); % Decomposição LU da matriz A
   coeffs = solve_lu(L, U, b);   % Resolve o sistema para encontrar os coeficientes

   % Garante que os coeficientes sejam um vetor linha
   coeffs = coeffs.'; % Transpõe os coeficientes para formato de linha
end



% Função: lu_decomposition
function [L, U] = lu_decomposition(A)
    n = size(A,1);  % Número de linhas de A
    L = eye(n);     % Matriz L inicializada como matriz identidade
    U = A;          % U é inicializada como a própria matriz A
    for k = 1:n-1
        for i = k+1:n
            if U(k,k) == 0
                error('Zero pivot encountered.'); % Verifica se o pivô é zero
            end
            L(i,k) = U(i,k) / U(k,k);    % Calcula o fator de escala para L
            U(i,:) = U(i,:) - L(i,k) * U(k,:); % Atualiza as linhas de U
        end
    end
end



% Função: solve_lu
function x = solve_lu(L, U, b)
    % Substituição para frente para resolver Ly = b
    n = length(b);         % Tamanho do vetor b
    y = zeros(n, 1);       % Inicializa o vetor y com zeros
    for i = 1:n
        % Calcula cada elemento de y usando os valores de L e b
        y(i) = (b(i) - L(i, 1:i-1) * y(1:i-1)) / L(i, i);
    end



    % Substituição para trás para resolver Ux = y
    x = zeros(n, 1);  % Inicializa o vetor x com zeros
    for i = n:-1:1    % Começa de trás para frente
        % Calcula cada elemento de x usando os valores de U e y
        x(i) = (y(i) - U(i, i+1:n) * x(i+1:n)) / U(i, i);
    end
end


% Função: filter_outliers_multivariate
function [filtered_data] = filter_outliers_multivariate(data)
    % 'data' é uma matriz onde cada coluna representa uma variável
    num_vars = size(data, 2);  % Número de variáveis (colunas)
    lower_bounds = zeros(1, num_vars); % Inicializa os limites inferiores
    upper_bounds = zeros(1, num_vars); % Inicializa os limites superiores

    % Calcula os limites para cada variável usando o intervalo interquartil (IQR)
    for i = 1:num_vars
        Q1 = prctile(data(:,i), 25);  % Primeiro quartil (25%)
        Q3 = prctile(data(:,i), 75);  % Terceiro quartil (75%)
        IQR = Q3 - Q1;                % Intervalo interquartil (IQR)
        lower_bounds(i) = Q1 - 1.5 * IQR;  % Limite inferior
        upper_bounds(i) = Q3 + 1.5 * IQR;  % Limite superior
    end

    % Filtra as linhas que estão dentro dos limites calculados para todas as variáveis
    valid_indices = true(size(data,1),1); % Inicializa uma máscara lógica
    for i = 1:num_vars
        valid_indices = valid_indices & data(:,i) >= lower_bounds(i) & data(:,i) <= upper_bounds(i);
    end

    % Retorna os dados filtrados (sem outliers)
    filtered_data = data(valid_indices, :);
end



% Função: quality_of_fit
function [Sr, r2, sy_x] = quality_of_fit(y, y_pred)
    % Sr: Soma dos quadrados dos resíduos (diferença entre y e y_pred)
    Sr = sum((y - y_pred).^2);

    % St: Soma total dos quadrados (diferença entre y e sua média)
    St = sum((y - mean(y)).^2);

    % r²: Coeficiente de determinação, que mede a qualidade do ajuste
    r2 = 1 - (Sr / St);

    % sy_x: Erro padrão da estimativa
    sy_x = sqrt(Sr / (length(y) - 2));  % Calcula o erro padrão da estimativa
end



% Nomes das colunas do conjunto de dados
columns_names = {'pelvic_incidence', 'pelvic_tilt', ...
                 'lumbar_lordosis_angle', 'sacral_slope', ...
                 'pelvic_radius', 'degree_spondylolisthesis', 'class'};

% Carrega os dados de um arquivo CSV, ignorando a primeira linha (cabeçalho)
data = csvread('column_2C_cleaned.csv', 1, 0);

% Filtra os dados de acordo com a classe
class_normal = data(data(:, end) == 0, :); % Dados da classe "normal"
class_anormal = data(data(:, end) == 1, :); % Dados da classe "anormal"

% Índice da coluna para a variável 'degree_spondylolisthesis'
degree_index = 6;

% Colunas selecionadas para análise
selected_columns = [1, 2, 3, 4, 5]; % Índices das colunas a serem usadas



% Função para filtrar outliers usando o método do intervalo interquartil (IQR)
function [filtered_x, filtered_y] = filter_outliers(x, y)
    % Calcula o primeiro e terceiro quartil de x
    Q1_x = prctile(x, 25); % Primeiro quartil para x
    Q3_x = prctile(x, 75); % Terceiro quartil para x
    IQR_x = Q3_x - Q1_x;   % Intervalo interquartil para x

    % Calcula o primeiro e terceiro quartil de y
    Q1_y = prctile(y, 25); % Primeiro quartil para y
    Q3_y = prctile(y, 75); % Terceiro quartil para y
    IQR_y = Q3_y - Q1_y;   % Intervalo interquartil para y



    % Define os limites inferior e superior para a filtragem com base no IQR
    lower_bound_x = Q1_x - 1.5 * IQR_x; % Limite inferior para x
    upper_bound_x = Q3_x + 1.5 * IQR_x; % Limite superior para x

    lower_bound_y = Q1_y - 1.5 * IQR_y; % Limite inferior para y
    upper_bound_y = Q3_y + 1.5 * IQR_y; % Limite superior para y

    % Filtra os dados, mantendo apenas os valores dentro dos limites calculados
    valid_indices = x >= lower_bound_x & x <= upper_bound_x & y >= lower_bound_y & y <= upper_bound_y;

    % Aplica a filtragem aos dados de x e y
    filtered_x = x(valid_indices);
    filtered_y = y(valid_indices);
end


% Gráfico de dispersão e análise de regressão
figure;
hold on; % Permite sobrepor gráficos

% Inicializa listas para armazenar r² e r para classes normal e anormal
r2_normal_list = [];
r_normal_list = [];
r2_abnormal_list = [];
r_abnormal_list = [];

% Inicializa listas para armazenar os dados filtrados
total_x_anormal_filtered = [];
total_x_normal_filtered = [];
total_y_anormal_filtered = [];
total_y_normal_filtered = [];

% Loop através das colunas selecionadas
for ii = 1:length(selected_columns)
    % Obtém os dados da classe anormal para a coluna atual
    x_anormal = class_anormal(:, ii);
    y_anormal = class_anormal(:, degree_index);

    % Obtém os dados da classe normal para a coluna atual
    x_normal = class_normal(:, ii);
    y_normal = class_normal(:, degree_index);

    % Filtra os outliers para a classe anormal
    [x_anormal_filtered, y_anormal_filtered] = filter_outliers(x_anormal, y_anormal);

    % Filtra os outliers para a classe normal
    [x_normal_filtered, y_normal_filtered] = filter_outliers(x_normal, y_normal);

    % Armazena os dados filtrados para futuras análises
    total_x_anormal_filtered = [total_x_anormal_filtered; x_anormal_filtered];
    total_x_normal_filtered = [total_x_normal_filtered; x_normal_filtered];
    total_y_anormal_filtered = [total_y_anormal_filtered; y_anormal_filtered];
    total_y_normal_filtered = [total_y_normal_filtered; y_normal_filtered];

    % Cria um subplot para a coluna atual
    subplot(2, 3, ii);
    hold on; % Permite adicionar mais gráficos no mesmo subplot

    % Gráfico de dispersão para os dados filtrados (classe normal e anormal)
    scatter(x_normal_filtered, y_normal_filtered, 'b', 'x'); % Classe normal
    scatter(x_anormal_filtered, y_anormal_filtered, 'r', 'o'); % Classe anormal
    xlabel(columns_names{ii}); % Nome da variável x
    ylabel(columns_names{degree_index}); % Nome da variável y
    title(['Scatter plot: ', columns_names{ii}, ' vs ', columns_names{degree_index}]); % Título do gráfico
    grid on; % Exibe a grade no gráfico


#########  RESPOSTA 1.1: #########
#A partir do gráfico, com a regressão linear, foi possível visualizar
#quais linhas de regressão se ajustaram melhor aos dados. Nesse sentido, as duas
#variáveis que mais se relacionam ao grau de espondilolistese são a Pelvic
#Incidence e Lumbar Lordosis Angle. Para verificar a correlação, plotamos um mapa
#de calor e nossa escolha foi corroborada.


### COMEÇO 1.2 ###

        % Regressão e cálculo de R² para a classe normal filtrada
    x = x_normal_filtered;
    y = y_normal_filtered;

    % Calcula os coeficientes da regressão linear para os dados normais filtrados
    [a, b] = linear_regression(x, y);

    % Calcula o coeficiente de determinação (R²) para a classe normal
    r2_normal = calculate_pearson(x, y);
    r_normal = sqrt(r2_normal); % Coeficiente de correlação (raiz de R²)

    % Armazena R² e r nas listas correspondentes
    r2_normal_list = [r2_normal_list; r2_normal];
    r_normal_list = [r_normal_list; r_normal];

    % Plota a linha de regressão para os dados normais
    plot(x, a + b * x, '-b', 'LineWidth', 1.5); % Linha azul

    % Regressão e cálculo de R² para a classe anormal filtrada
    x_abnormal = x_anormal_filtered;
    y_abnormal = y_anormal_filtered;

    % Calcula os coeficientes da regressão linear para os dados anormais filtrados
    [a_ab, b_ab] = linear_regression(x_abnormal, y_abnormal);

    % Calcula o coeficiente de determinação (R²) para a classe anormal
    r2_abnormal = calculate_pearson(x_abnormal, y_abnormal);
    r_abnormal = sqrt(r2_abnormal); % Coeficiente de correlação

    % Armazena R² e r nas listas correspondentes
    r2_abnormal_list = [r2_abnormal_list; r2_abnormal];
    r_abnormal_list = [r_abnormal_list; r_abnormal];

    % Plota a linha de regressão para os dados anormais
    plot(x_abnormal, a_ab + b_ab * x_abnormal, '-r', 'LineWidth', 1.5); % Linha vermelha

    hold off; % Finaliza o subplot atual
end


% MAPA DE CALOR
% Calcula a matriz de correlação para as variáveis (excluindo a última coluna "class")
correlation_matrix = corr(data(:, 1:end-1));

% Cria a figura para o mapa de calor
figure(2);
imagesc(correlation_matrix); % Gera o mapa de calor com base na matriz de correlação
colorbar; % Adiciona uma barra de cores para referência

% Ajusta a colormap para 'jet' (colormap padrão com variação de cores)
colormap(jet);

% Configura os rótulos dos eixos com os nomes das variáveis
set(gca, 'XTick', 1:length(columns_names)-1, 'XTickLabel', columns_names(1:end-1), 'XTickLabelRotation', 45); % Rotaciona os rótulos do eixo X
set(gca, 'YTick', 1:length(columns_names)-1, 'YTickLabel', columns_names(1:end-1)); % Rótulos do eixo Y

% Define o título e os rótulos dos eixos
title('Correlogram of Variables');
xlabel('Variables');
ylabel('Variables');

% Adiciona os valores numéricos de correlação dentro do mapa de calor
for i = 1:size(correlation_matrix, 1)
    for j = 1:size(correlation_matrix, 2)
        % Escolhe a cor do texto com base na intensidade da correlação
        if abs(correlation_matrix(i, j)) > 0.5
            text_color = 'w'; % Branco para correlações altas
        else
            text_color = 'k'; % Preto para correlações baixas
        end
        % Exibe os valores da correlação dentro do mapa
        text(j, i, num2str(correlation_matrix(i, j), '%.2f'), ...
            'HorizontalAlignment', 'center', 'Color', text_color);
    end
end



######### RESPOSTA 1.2:#########
# Por meio do cálculo do R2 , para classe normal (3.5656e-02; 5.6541e-02; 2.4114e-02 ;
# 1.9073e-03 ; 1.7731e-04 ) e anormal (0.459783;  0.189754; 0.493209;
# 0.253549;  0.036848) representa a variação do grau de espondilolistese. Além #disso, quanto maior o coeficiente de regressão maior será a influência da #característica. A matriz de correlação foi calculada através do mapa de calor, #quanto mais próximo de 1 ou -1, as variáveis têm uma relação mais forte com o #grau de espondilolistese, neste casos são Pelvic Incidence (0,64) e Lumbar #Lordosis Angle (0,53).





######### ANÁLISE 2 #########
 ######### COMEÇO 2.1 #########

% Exibe os valores de R² e R para as classes normal e anormal
disp('R² values for normal class:');
disp(r2_normal_list);
disp('R² values for abnormal class:');
disp(r2_abnormal_list);
disp('R values for normal class:');
disp(r_normal_list);
disp('R values for abnormal class:');
disp(r_abnormal_list);

% Inicializa mapas (dicionários) para armazenar valores de R e R² para cada variável
r2_normal_map = containers.Map('KeyType', 'double', 'ValueType', 'any');
r_normal_map = containers.Map('KeyType', 'double', 'ValueType', 'any');
r2_abnormal_map = containers.Map('KeyType', 'double', 'ValueType', 'any');
r_abnormal_map = containers.Map('KeyType', 'double', 'ValueType', 'any');

% Inicializa variáveis para armazenar os dados filtrados (x e y) de ambas as classes
total_x_anormal_filtered = [];
total_x_normal_filtered = [];
total_y_anormal_filtered = [];
total_y_normal_filtered = [];

% Cria uma nova figura para os gráficos de dispersão
figure(3)
for ii = 1:length(selected_columns)
    % Extrai os dados da classe anormal e normal para a coluna atual
    x_anormal = class_anormal(:, ii);
    y_anormal = class_anormal(:, degree_index);
    x_normal = class_normal(:, ii);
    y_normal = class_normal(:, degree_index);

    % Filtra os outliers para ambas as classes
    [x_anormal_filtered, y_anormal_filtered] = filter_outliers(x_anormal, y_anormal);
    [x_normal_filtered, y_normal_filtered] = filter_outliers(x_normal, y_normal);

    % Armazena os dados filtrados para análises futuras
    total_x_anormal_filtered = [total_x_anormal_filtered; x_anormal_filtered];
    total_x_normal_filtered = [total_x_normal_filtered; x_normal_filtered];
    total_y_anormal_filtered = [total_y_anormal_filtered; y_anormal_filtered];
    total_y_normal_filtered = [total_y_normal_filtered; y_normal_filtered];

    % Inicializa o subplot para o gráfico da variável atual
    subplot(2, 3, ii);
    hold on; % Permite sobrepor gráficos

    % Cria gráficos de dispersão para os dados normais e anormais filtrados
    scatter(x_normal_filtered, y_normal_filtered, 'b', 'x'); % Classe normal (azul)
    scatter(x_anormal_filtered, y_anormal_filtered, 'r', 'o'); % Classe anormal (vermelho)

    % Define os rótulos e o título do gráfico
    xlabel(columns_names{ii}); % Nome da variável no eixo X
    ylabel(columns_names{degree_index}); % Nome da variável no eixo Y
    title(['Scatter plot: ', columns_names{ii}, ' vs ', columns_names{degree_index}]); % Título
    grid on; % Exibe a grade no gráfico


######### RESPOSTA 2.1:#########
#A melhor candidata para uma regressão polinomial é a pelvic #radius, isto é, a que apresentou comportamento menos linear

### COMEÇO 2.2 ###
        % Define cores para as regressões de grau 1, 2 e 3
    cores = {'-b', '-g', '-r'}; % Azul para grau 1, verde para grau 2, vermelho para grau 3

    % Regressão e R² para a classe normal
    x = x_normal_filtered; % Dados filtrados da classe normal (variável independente)
    y = y_normal_filtered; % Dados filtrados da classe normal (variável dependente)

    % Loop para aplicar regressões polinomiais de grau 1, 2 e 3
    for grau = 1:3
        % Calcula os coeficientes da regressão polinomial para o grau atual
        coeffs_normal = regressao_polinomial(x, y, grau);

        % Gera um intervalo de valores de x para plotar a curva polinomial
        x_range = linspace(min(x), max(x), 100);

        % Avalia o polinômio nos valores de x_range
        y_poly_normal = polyval(coeffs_normal, x_range);

        % Plota a curva de regressão polinomial para o grau atual com a cor correspondente
        plot(x_range, y_poly_normal, cores{grau}, 'LineWidth', 1.5);

        % Calcula os valores previstos (y_pred) com o polinômio para os dados de x
        y_pred_normal = polyval(coeffs_normal, x);

        % Calcula a soma dos quadrados dos resíduos (ss_res) e a soma total dos quadrados (ss_tot)
        ss_res_normal = sum((y - y_pred_normal).^2);
        ss_tot_normal = sum((y - mean(y)).^2);

        % Calcula o coeficiente de determinação (R²) para a classe normal
        r2 = 1 - (ss_res_normal / ss_tot_normal);
        r = sqrt(r2); % Calcula o coeficiente de correlação (raiz quadrada de R²)

        % Armazena os valores de R² e R nos dicionários, agrupando por grau
        if isKey(r2_normal_map, grau)
            r2_normal_map(grau) = [r2_normal_map(grau), r2];
            r_normal_map(grau) = [r_normal_map(grau), r];
        else
            r2_normal_map(grau) = r2;
            r_normal_map(grau) = r;
        end

        % Exibe os resultados no console
        fprintf('Normal Class | Column: %s | Degree: %d | R: %.4f | R²: %.4f\n', columns_names{ii}, grau, r, r2);
    end


######### RESPOSTA 2.2:#########
# Gráfico plotado


## COMEÇO 2.3 ##
    % Regressão e R² para a classe anormal
x_abnormal = x_anormal_filtered; % Dados filtrados da classe anormal (variável independente)
y_abnormal = y_anormal_filtered; % Dados filtrados da classe anormal (variável dependente)

% Loop para aplicar regressões polinomiais de grau 1, 2 e 3
for grau = 1:3
    % Calcula os coeficientes da regressão polinomial para o grau atual
    coeffs_anormal = regressao_polinomial(x_abnormal, y_abnormal, grau);

    % Gera um intervalo de valores de x para plotar a curva polinomial
    x_range = linspace(min(x_abnormal), max(x_abnormal), 100);

    % Avalia o polinômio nos valores de x_range
    y_poly_anormal = polyval(coeffs_anormal, x_range);

    % Plota a curva de regressão polinomial para o grau atual com a cor correspondente
    plot(x_range, y_poly_anormal, cores{grau}, 'LineWidth', 1.5);

    % Calcula os valores previstos (y_pred) com o polinômio para os dados de x
    y_pred_abnormal = polyval(coeffs_anormal, x_abnormal);

    % Calcula a soma dos quadrados dos resíduos (ss_res) e a soma total dos quadrados (ss_tot)
    ss_res_abnormal = sum((y_abnormal - y_pred_abnormal).^2);
    ss_tot_abnormal = sum((y_abnormal - mean(y_abnormal)).^2);

    % Calcula o coeficiente de determinação (R²) para a classe anormal
    r2_ab = 1 - (ss_res_abnormal / ss_tot_abnormal);
    r_ab = sqrt(r2_ab); % Calcula o coeficiente de correlação (raiz quadrada de R²)

    % Armazena os valores de R² e R nos dicionários, agrupando por grau
    if isKey(r2_abnormal_map, grau)
        r2_abnormal_map(grau) = [r2_abnormal_map(grau), r2_ab];
        r_abnormal_map(grau) = [r_abnormal_map(grau), r_ab];
    else
        r2_abnormal_map(grau) = r2_ab;
        r_abnormal_map(grau) = r_ab;
    end


   % Exibe os resultados da regressão polinomial para a classe anormal
   % Esta parte imprime o grau do polinômio, os valores de R (correlação) e R² (coeficiente de determinação)
   fprintf('Abnormal Class | Column: %s | Degree: %d | R: %.4f | R²: %.4f\n', columns_names{ii}, grau, r_ab, r2_ab);
end

% Finaliza o subplot após plotar as curvas para os três graus de polinômio
hold off;
end

% Exibe os valores armazenados nos dicionários, começando pela classe normal
% Isso fornece uma visão geral dos valores de R² para cada grau de polinômio

fprintf('\nR² values for Normal Class:\n'); % Cabeçalho para R² da classe normal
keys_normal = keys(r2_normal_map); % Obtém as chaves (graus de polinômio) no dicionário
for i = 1:length(keys_normal)
    grau = keys_normal{i}; % Pega o grau atual
    values_r2_normal = r2_normal_map(grau); % Pega os valores de R² correspondentes ao grau atual
    fprintf('Degree %d: R² values = [', grau); % Exibe o grau atual
    fprintf(' %.4f', values_r2_normal); % Exibe cada valor de R² no formato de quatro casas decimais
    fprintf(' ]\n'); % Fecha a linha para aquele grau
end


% Exibe os valores de R (correlação) para a classe normal
fprintf('\nR values for Normal Class:\n');
for i = 1:length(keys_normal)
    grau = keys_normal{i}; % Pega o grau atual
    values_r_normal = r_normal_map(grau); % Pega os valores de R correspondentes ao grau atual
    fprintf('Degree %d: R values = [', grau); % Exibe o grau atual
    fprintf(' %.4f', values_r_normal); % Exibe cada valor de R no formato de quatro casas decimais
    fprintf(' ]\n'); % Fecha a linha para aquele grau
end


% Exibe os valores de R² para a classe anormal, de forma semelhante à exibição para a classe normal
fprintf('\nR² values for Abnormal Class:\n');
keys_abnormal = keys(r2_abnormal_map); % Obtém as chaves (graus de polinômio) para a classe anormal
for i = 1:length(keys_abnormal)
    grau = keys_abnormal{i}; % Pega o grau atual
    values_r2_abnormal = r2_abnormal_map(grau); % Pega os valores de R² correspondentes ao grau atual
    fprintf('Degree %d: R² values = [', grau); % Exibe o grau atual
    fprintf(' %.4f', values_r2_abnormal); % Exibe cada valor de R² no formato de quatro casas decimais
    fprintf(' ]\n'); % Fecha a linha para aquele grau
end


% Exibe os valores de R (correlação) para a classe anormal
fprintf('\nR values for Abnormal Class:\n');
for i = 1:length(keys_abnormal)
    grau = keys_abnormal{i}; % Pega o grau atual
    values_r_abnormal = r_abnormal_map(grau); % Pega os valores de R correspondentes ao grau atual
    fprintf('Degree %d: R values = [', grau); % Exibe o grau atual
    fprintf(' %.4f', values_r_abnormal); % Exibe cada valor de R no formato de quatro casas decimais
    fprintf(' ]\n'); % Fecha a linha para aquele grau
end

######### RESPOSTA 2.3: #########
#Segundo R e R^2, chegamos a conclusão que o polinômio 3 seria o melhor para o conjunto de dados.

######### ANÁLISE 3 #########
### COMEÇO 3.1 ###
% Carregar os dados do arquivo CSV
data_struct = importdata('column_2C_cleaned.csv');
data = data_struct.data;  % Extrai os dados numéricos

% Definir as variáveis conforme as colunas do dataset
x1 = data(:, 1);  % "pelvic_incidence"
x2 = data(:, 3);  % "lumbar_lordosis_angle"
y  = data(:, 6);  % "degree_spondylolisthesis"

% --- Modelo 1: Regressão Linear Simples com x1 ---

% Filtrar outliers para x1 e y
data_model1 = [x1, y];
filtered_data_model1 = filter_outliers_multivariate(data_model1);
filtered_x1 = filtered_data_model1(:,1);
filtered_y1 = filtered_data_model1(:,2);

% Calcular os coeficientes e valores previstos para o modelo 1
[a1, b1] = linear_regression(filtered_x1, filtered_y1);
y_pred1 = a1 + b1 .* filtered_x1;
[Sr1, r21, sy_x1] = quality_of_fit(filtered_y1, y_pred1);

% Exibir resultados do modelo 1
fprintf('Resultados do Modelo 1:\n');
fprintf('Intercepto (a1): %f, Inclinação (b1): %f, Sr1: %f, r²: %f, sy/x: %f\n\n', a1, b1, Sr1, r21, sy_x1);

% --- Modelo 2: Regressão Linear Simples com x2 ---

% Filtrar outliers para x2 e y
data_model2 = [x2, y];
filtered_data_model2 = filter_outliers_multivariate(data_model2);
filtered_x2 = filtered_data_model2(:,1);
filtered_y2 = filtered_data_model2(:,2);

% Calcular os coeficientes e valores previstos para o modelo 2
[a2, b2] = linear_regression(filtered_x2, filtered_y2);
y_pred2 = a2 + b2 .* filtered_x2;
[Sr2, r22, sy_x2] = quality_of_fit(filtered_y2, y_pred2);

% Exibir resultados do modelo 2
fprintf('Resultados do Modelo 2:\n');
fprintf('Intercepto (a2): %f, Inclinação (b2): %f, Sr2: %f, r²: %f, sy/x: %f\n\n', a2, b2, Sr2, r22, sy_x2);


######### RESPOSTA 3.1:#########
#|Após o cálculo das métricas desejadas conclui-se que o
#modelo 2 foi o que apresentou o maior ajuste, uma vez que, seu Sy/x < que o
#encontrado no modelo 1.  R2 modelo 2 > R2 modelo 1 e Sr2<Sr1 indicando um
#resíduo menor no  modelo  escolhido

#A métrica de Sy/x indica o erro padrão da estimativa que mede a dispersão dos
#pontos dada uma reta de regressão em relação aos valores de x e y -Quanto
#menor seu valor mais preciso o modelo é, uma vez que, indica a proximidade com
#a regressão

#A métrica R2 indica o quanto o modelo se ajusta aos dados. Isso significa, caso R2
#seja alto, que a variável independente explica uma maior parte da variação da
#variável dependente.

#MODELO 1: Intercepto (a1): -42.137502, Inclinaթ«թ։o (b1): 1.075177,
#Sr1: 100885.351328, r^2: 0.470416, sy/x: 18.461552
#MODELO 2:  Intercepto (a2): -29.489106, Inclinaթ«թ։o (b2): 1.007476,
#Sr2: 97522.583616, r^2: 0.504229, sy/x: 18.090246



% --- Modelo 3: Regressão Linear Múltipla com x1 e x2 ---

#### COMEÇO 3.2 ####
% Filtrar outliers para x1, x2 e y (regressão múltipla)
data_model3 = [x1, x2, y];  % Combina x1, x2 e y em uma matriz
filtered_data_model3 = filter_outliers_multivariate(data_model3);  % Filtra outliers
filtered_x1_multi = filtered_data_model3(:,1);  % Dados filtrados para x1
filtered_x2_multi = filtered_data_model3(:,2);  % Dados filtrados para x2
filtered_y_multi = filtered_data_model3(:,3);   % Dados filtrados para y

% Criar a matriz de design para a regressão linear múltipla
X_multi = [ones(length(filtered_y_multi),1), filtered_x1_multi, filtered_x2_multi];  % Adiciona a constante
XtX_multi = X_multi' * X_multi;  % Produto da transposta de X por X
Xty_multi = X_multi' * filtered_y_multi;  % Produto da transposta de X por y

% Decomposição LU para resolver o sistema linear
[L, U] = lu_decomposition(XtX_multi);  % Decomposição LU da matriz XtX
theta = solve_lu(L, U, Xty_multi);  % Resolve o sistema linear para encontrar os coeficientes
a0 = theta(1);  % Intercepto
a1_multi = theta(2);  % Coeficiente para x1
a2_multi = theta(3);  % Coeficiente para x2

% Calcular os valores previstos para y usando o modelo de regressão múltipla
y_pred3 = X_multi * theta;  % Predição de y com base no modelo

% Calcular Sr (soma dos quadrados dos resíduos) e St (soma total dos quadrados)
Sr3 = sum((filtered_y_multi - y_pred3).^2);  % Soma dos quadrados dos resíduos
St3 = sum((filtered_y_multi - mean(filtered_y_multi)).^2);  % Soma total dos quadrados

% Calcular r² e o erro padrão sy/x
r2_3 = 1 - (Sr3 / St3);  % Coeficiente de determinação r²
sy_x3 = sqrt(Sr3 / (length(filtered_y_multi) - 3));  % Erro padrão da estimativa

% Exibir os resultados do modelo de regressão múltipla
fprintf('Resultados do Modelo 3 (Regressão Linear Múltipla):\n');
fprintf('Intercepto (a0): %f, Coeficientes (a1, a2): %f, %f, Sr3: %f, r²: %f, sy/x: %f\n\n', a0, a1_multi, a2_multi, Sr3, r2_3, sy_x3);


######### RESPOSTA 3.2: #########
# os resultados do modelo 3 são: a0 = -41.6348, a1 = 0.5182, a2 = 0.6371; o
# 𝑆𝑟 = 97522.5836 , 𝑟2 = 0.55100 e 𝑠𝑦/𝑥 = 17.0278
# Concluímos que o modelo 3 é o melhor. devido ao Sr e 𝑠𝑦/𝑥  menor, e R2 mais
# próximo de 1


### COMEÇO 3.3 ####
% --- Comparação dos Modelos ---

fprintf('\nComparação dos Modelos:\n');
% Exibir os resultados de Sr, r² e sy/x para os três modelos
fprintf('Modelo 1: Sr = %.4f, r² = %.4f, syx = %.4f\n', Sr1, r21, sy_x1);  % Resultados do Modelo 1 (x1)
fprintf('Modelo 2: Sr = %.4f, r² = %.4f, syx = %.4f\n', Sr2, r22, sy_x2);  % Resultados do Modelo 2 (x2)
fprintf('Modelo 3: Sr = %.4f, r² = %.4f, syx = %.4f\n', Sr3, r2_3, sy_x3); % Resultados do Modelo 3 (regressão múltipla)

% Decisão sobre o melhor modelo com base no valor de r²
if r2_3 > r21 && r2_3 > r22
    % Se o modelo de regressão múltipla (Modelo 3) tiver o maior r²
    fprintf('O modelo de regressão múltipla (Modelo 3) forneceu o melhor ajuste.\n');
else
    % Caso contrário, indica que os modelos simples foram melhores ou equivalentes
    fprintf('O modelo de regressão múltipla não superou os modelos mais simples.\n');
end


######### RESPOSTA 3.3 #########
# apesar de o modelo 3 ser o melhor entre eles, nenhum dos modelos pode ser
# definido como ideal, visto que seus valores de R2 são abaixo de 0,6
# isso significa que nenhum dos modelos explica mais do que 60% da variabilidade
# dos dados
# além disso, os valores de Sr e 𝑠𝑦/𝑥 são muito altos, o que mostra que as previsões
# estão distantes de previsões reais

