pasta = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'simulador', 'ensaios', 'dados', 'banguela_20260917_10ms');
serie = jsondecode(fileread(fullfile(pasta, 'serie.json')));
ajuste = jsondecode(fileread(fullfile(pasta, 'identificacao.json')));
ganho = ajuste.gain_motor_normalized;
atrito = ajuste.coast_loss.c0_m_s2;
treino = {};
validacao = {};

assert(ajuste.coast_loss.c1_s_inv == 0 && ...
    ajuste.coast_loss.c2_m_inv == 0, 'O modelo exige atrito constante.');

for k = 1:numel(serie.trials)
    ensaio = serie.trials(k);
    assert(ensaio.complete, 'Ensaio incompleto.');
    D = readtable(fullfile(pasta, ensaio.file));
    indices = find(D.t >= 0.3 & D.v_raw >= 0.45);
    D = D(indices(1):indices(end), :);
    dados = [D.t, ganho * D.u_motor - atrito, D.v_raw];
    if ensaio.repetition < serie.repetitions
        treino{end+1} = dados;
    else
        validacao{end+1} = dados;
    end
end

tau = fminbnd(@(tau) erro(tau, treino), 1e-4, 1);
G = tf(1, [tau 1 0])
fprintf('tau = %.6f s\n', tau);
fprintf('RMSE de validacao = %.6f m/s\n', sqrt(erro(tau, validacao)));
if tau <= serie.sample_time_s
    fprintf('tau esta no limite da resolucao temporal dos ensaios.\n');
end

function mse = erro(tau, ensaios)
    soma = 0;
    n = 0;
    for k = 1:numel(ensaios)
        D = ensaios{k};
        t = D(:, 1);
        u = D(:, 2);
        v = D(1, 3);
        a = u(1);
        for j = 1:numel(t)-1
            dt = t(j+1) - t(j);
            h = -expm1(-dt / tau);
            v = v + u(j) * dt + (a - u(j)) * tau * h;
            a = a + (u(j) - a) * h;
            soma = soma + (v - D(j+1, 3))^2;
            n = n + 1;
        end
    end
    mse = soma / n;
end
