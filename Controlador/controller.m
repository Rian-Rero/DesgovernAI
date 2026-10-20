%% Planta
run(fullfile(fileparts(mfilename('fullpath')), 'planta.m'));
s = tf('s');
figure('Name', 'LGR da planta');
rlocus(G); grid on; title('Lugar das raizes da planta G(s)');

figure('Name', 'Bode da planta');
bode(G); grid on; title('Bode da planta');

%% Restricoes
Ts = serie.sample_time_s;
sobressinal_max = 5;
margem_fase_min = 60;
margem_ganho_min = 6;
tempo = (0:Ts:20)';
velocidades = serie.speeds(:)';
delta_v = max(velocidades) - min(velocidades);
incrementos = velocidades(2:end) - min(velocidades);
transicoes = [incrementos -incrementos];
parametros = serie.car_parameters;
k_motor = 1.26 * ganho;
compensacao = parametros.MI * parametros.GRAV;
u_min = k_motor * (-parametros.ACCELMAX + compensacao) - atrito;
u_max = k_motor * (parametros.ACCELMAX + compensacao) - atrito;
Gd = c2d(G, Ts, 'zoh');

%% Contas de projeto
fprintf('\nC(s) = (kp*s + ki)/s\n');
fprintf('Polinomio: tau*s^3 + s^2 + kp*s + ki\n');
fprintf('Routh: kp > tau*ki, kp > 0, ki > 0\n');
fprintf('Polos desejados: (s+p)*(s^2+2*zeta*wn*s+wn^2)\n');
fprintf('p = 1/tau - 2*zeta*wn\n');
fprintf('kp = tau*(wn^2 + 2*zeta*wn*p)\n');
fprintf('ki = tau*wn^2*p\n');
fprintf('Limites da entrada normalizada: [%.4f, %.4f] m/s^2\n', u_min, u_max);

amortecimentos = [0.7 0.9 1 1.4 2 3];
frequencias = logspace(-1, log10(1/(8*tau)), 70);
resultados = [];
melhor_tempo = Inf;

for zeta = amortecimentos
    for wn = frequencias
        p = 1/tau - 2*zeta*wn;
        if p <= 0
            continue
        end
        kp_teste = tau * (wn^2 + 2*zeta*wn*p);
        ki_teste = tau * wn^2 * p;
        Cteste = (kp_teste*s + ki_teste)/s;
        Ld = c2d(Cteste, Ts, 'zoh') * Gd;
        [gm, pm] = margin(Ld);
        if ~isstable(feedback(Ld, 1)) || pm < margem_fase_min || ...
                20*log10(gm) < margem_ganho_min
            continue
        end
        tempos = zeros(size(transicoes));
        picos = zeros(size(transicoes));
        for j = 1:numel(transicoes)
            referencia = transicoes(j);
            y = simular_pi(kp_teste, ki_teste, tau, tempo, referencia, u_min, u_max);
            [tempos(j), picos(j)] = desempenho(y, referencia, Ts);
        end
        ta = max(tempos);
        pico = max(picos);
        resultados(end+1, :) = [kp_teste ki_teste ta pico pm wn zeta];
        if pico <= sobressinal_max && ta < melhor_tempo
            melhor_tempo = ta;
            kp = kp_teste;
            ki = ki_teste;
            wn_projeto = wn;
            zeta_projeto = zeta;
        end
    end
end
assert(isfinite(melhor_tempo), 'Nenhum PI satisfez as restricoes da busca.');

%% PI aplicado e validado no CoppeliaSim
fprintf('Busca nominal inicial: kp = %.6f; ki = %.6f\n', kp, ki);
arquivo = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'simulador', 'fva_car', 'speed_controller.py');
configuracao = fileread(arquivo);
kp = parametro(configuracao, 'SPEED_KP');
ki = parametro(configuracao, 'SPEED_KI');
Ts = parametro(configuracao, 'CONTROL_SAMPLE_TIME');
Tf = parametro(configuracao, 'VELOCITY_FILTER_TIME');
Tu = parametro(configuracao, 'CONTROL_FILTER_TIME');
tempo = (0:Ts:20)';
H = 1 / (Tf*s + 1);
alpha = -expm1(-Ts/Tf);
Hd = tf([alpha 0], [1 -(1-alpha)], Ts);
Hu = 1 / (Tu*s + 1);
alpha_u = -expm1(-Ts/Tu);
Hud = tf([alpha_u 0], [1 -(1-alpha_u)], Ts);
Gd = c2d(G, Ts, 'zoh');
melhor_tempo = 0;
for incremento = transicoes
    y = simular_pi(kp, ki, tau, tempo, incremento, u_min, u_max, Tf, Tu);
    ta = desempenho(y, incremento, Ts);
    melhor_tempo = max(melhor_tempo, ta);
end

%% Controlador e malha fechada
C = (kp*s + ki)/s;
L = C * Hu * G * H;
T = feedback(C*Hu*G, H);
Tc = feedback(L, 1);
S = feedback(1, L);
CS = minreal(Hu * C * S);
Cd = c2d(C, Ts, 'zoh');
Ld = Cd * Hud * Gd * Hd;
Td = feedback(Cd*Hud*Gd, Hd);
assert(isstable(T) && isstable(Td) && kp > tau*ki);
erro_estacionario = dcgain(S);
assert(abs(erro_estacionario) < 1e-9, 'Erro estacionario diferente de zero.');
[gm, pm, ~, wc] = margin(Ld);
fprintf('\nkp = %.6f; ki = %.6f\n', kp, ki);
fprintf('Amostragem = %.3f s; filtros de velocidade/acao = %.3f/%.3f s\n', ...
    Ts, Tf, Tu);
fprintf('Erro estacionario para degrau = %.3g\n', erro_estacionario);
fprintf('Acomodacao de 2%%, pior transicao simulada = %.3f s\n', melhor_tempo);
fprintf('Margens digitais: %.2f dB, %.2f graus; wc = %.3f rad/s\n', ...
    20*log10(gm), pm, wc);
fprintf('Ganhos aplicados lidos do Python e validados na simulacao real.\n');
C
disp('Polos de malha fechada:'); disp(pole(T));

%% LGR, Bode e loop shaping
figure('Name', 'LGR com PI');
rlocus((s + ki/kp)/s * Hu * G * H); grid on; hold on;
plot(real(pole(T)), imag(pole(T)), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
title(sprintf('LGR com zero do PI; ganho escolhido kp = %.4f', kp));

figure('Name', 'Bode: planta e controlador');
bode(G, C, H, Hu, L); grid on;
legend('G', 'C', 'H velocidade', 'H acao', 'L = C H_u G H');

figure('Name', 'Loop shaping e margens');
margin(L); grid on; title('Loop shaping: L(s) = C(s)G(s)H(s)');
figure('Name', 'Margens com amostragem');
margin(Ld); grid on; title('Malha digital: controlador e planta com ZOH');

figure('Name', 'Sensibilidade');
bodemag(S, Tc); grid on; legend('S = 1/(1+L)', 'Tc = L/(1+L)');
title('Sensibilidade e sensibilidade complementar');
figure('Name', 'Esforco de controle');
bodemag(CS); grid on; title('C(s)S(s): referencia para entrada normalizada');

figure('Name', 'Resposta linear');
step(T, Td, 2); grid on; legend('Continuo', 'Digital');
title('Degrau unitario sem saturacao');
figure('Name', 'Rejeicao de perturbacao');
step(minreal(G*S), tempo); grid on;
title('Perturbacao degrau na entrada da planta: G(s)S(s)');

%% Comparacao dos projetos
viaveis = isfinite(resultados(:, 3)) & resultados(:, 4) <= sobressinal_max;
figure('Name', 'Busca do menor tempo');
scatter(resultados(viaveis, 6), resultados(viaveis, 3), 35, ...
    resultados(viaveis, 7), 'filled');
set(gca, 'XScale', 'log'); grid on; colorbar;
xlabel('wn (rad/s)'); ylabel('Acomodacao de 2% (s)');
title('Busca nominal inicial sem filtro; cor indica zeta');

%% Velocidades com saturacao e anti-windup
figure('Name', 'Referencias de velocidade');
tiledlayout(2, 1);
nexttile; hold on; grid on;
for vref = velocidades
    incremento = vref - min(velocidades);
    y = simular_pi(kp, ki, tau, tempo, incremento, u_min, u_max, Tf, Tu);
    plot(tempo, y + min(velocidades), 'DisplayName', sprintf('%.1f m/s', vref));
    yline(vref, ':', 'HandleVisibility', 'off');
    if incremento > 0
        [ta, pico] = desempenho(y, incremento, Ts);
        fprintf('Referencia %.1f m/s: acomodacao %.3f s; sobressinal %.2f%%\n', vref, ta, pico);
    end
end
xlabel('Tempo (s)'); ylabel('Velocidade (m/s)'); legend('Location', 'best');
title('Partida em 0.5 m/s, saturacao e anti-windup');
nexttile; hold on; grid on;
for sentido = [1 -1]
    [y, u] = simular_pi(kp, ki, tau, tempo, sentido*delta_v, u_min, u_max, Tf, Tu);
    [ta, pico] = desempenho(y, sentido*delta_v, Ts);
    fprintf('Transicao %+.1f m/s: acomodacao %.3f s; sobressinal %.2f%%\n', ...
        sentido*delta_v, ta, pico);
    comando_carro = (u + atrito)/k_motor - compensacao;
    plot(tempo, comando_carro, 'DisplayName', sprintf('Delta v = %+.1f m/s', sentido*delta_v));
end
yline(parametros.ACCELMAX, '--', 'HandleVisibility', 'off');
yline(-parametros.ACCELMAX, '--', 'HandleVisibility', 'off');
xlabel('Tempo (s)'); ylabel('Comando set\_u (m/s^2)'); legend;
title('Esforco limitado do carrinho');

function [y, u] = simular_pi(kp, ki, tau, t, referencia, u_min, u_max, Tf, Tu)
    y = zeros(size(t));
    u = zeros(size(t));
    integral = 0;
    a = 0;
    dt = t(2) - t(1);
    h = -expm1(-dt/tau);
    alpha_v = 1;
    alpha_u = 1;
    if nargin >= 8
        alpha_v = -expm1(-dt/Tf);
    end
    if nargin >= 9
        alpha_u = -expm1(-dt/Tu);
    end
    velocidade = 0;
    acao = 0;
    for k = 1:numel(t)-1
        velocidade = (1-alpha_v)*velocidade + alpha_v*y(k);
        e = referencia - velocidade;
        livre = kp*e + ki*integral;
        alvo = min(max(livre, u_min), u_max);
        if livre == alvo || (livre > u_max && e < 0) || (livre < u_min && e > 0)
            integral = integral + e*dt;
        end
        acao = (1-alpha_u)*acao + alpha_u*alvo;
        u(k) = acao;
        y(k+1) = y(k) + u(k)*dt + (a-u(k))*tau*h;
        a = a + (u(k)-a)*h;
    end
    u(end) = u(end-1);
end

function valor = parametro(codigo, nome)
    token = regexp(codigo, [nome '\s*=\s*([0-9.]+)'], 'tokens', 'once');
    assert(~isempty(token), 'Parametro %s nao encontrado.', nome);
    valor = str2double(token{1});
end

function [ta, pico] = desempenho(y, referencia, Ts)
    relativo = y / referencia;
    pico = max(0, 100*(max(relativo)-1));
    ultimo = find(abs(relativo-1) > 0.02, 1, 'last');
    ta = 0;
    if ~isempty(ultimo)
        ta = ultimo * Ts;
        if ultimo == numel(y)
            ta = Inf;
        end
    end
end
