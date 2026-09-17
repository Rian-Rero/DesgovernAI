# Ensaios de banguela e identificacao longitudinal

A serie padrao usa 0.5, 1.0, 1.5, 2.0 e 2.5 m/s, com tres repeticoes por
velocidade e amostragem de 10 ms. Cada ensaio recarrega a cena, acelera com
`set_vel(v0)`, confirma a velocidade medida por 1 s e passa para `set_coast()`.
O torque dos dois motores fica exatamente zero ate a parada. `set_u(0)` com
compensacao de atrito habilitada nao corresponde a banguela.

O coletor usa a pista de cones por padrao, partindo de (-24, -8.45) m e
interrompendo antes de X=21 m. A protecao lateral usa +/-1.4 m do centro;
ha tambem verificacao da altura e limite de duracao. Na opcao `--scene rampa`,
o trecho termina em X=-6 m, ANTES da inclinacao: dados em subida ou descida
nao devem ser misturados com dados planos para estimar as perdas.

## Executar

Com o CoppeliaSim aberto e a simulacao parada, na raiz do projeto:

```bash
venv/bin/python simulador/ensaio_banguela.py
venv/bin/python simulador/identificar_banguela.py simulador/ensaios/dados/PASTA_DA_SERIE
venv/bin/python -m unittest discover -s simulador/tests -v
```

`--port` escolhe a porta da API local; `--output` escolhe a pasta dos dados;
`--sample-time` permite 5, 10, 20 ou 50 ms. Para retomar uma serie interrompida,
use `--output PASTA --resume`, mantendo velocidades, repeticoes e amostragem.
Ensaios completos sao preservados. Tentativas incompletas sao arquivadas e
nao entram no ajuste. Nenhuma alteracao e salva nas cenas `.ttt`.

## Dados e modelo

Cada CSV separa `accelerate`, `hold` e `coast`, registrando velocidade
longitudinal bruta e filtrada, posicao, comando `u`, torque de cada motor e
`u_motor=(T_L+T_R)/(m*R)`. A velocidade bruta projeta a velocidade global no
eixo dianteiro do carro, sem confundir a queda vertical inicial com avancamento.
O comando registrado em t atua no intervalo [t, t_proximo).

Em banguela, o ajuste usa a equacao integrada de
`dv/dt=-(c0+c1*v+c2*v^2)`, sem diferenciar o sinal ruidoso.
Esse tipo de ensaio identifica a resistencia ao movimento, como no
[exemplo de coast-down da MathWorks](https://www.mathworks.com/help/sldo/ug/estimate-vehicle-drag-coefficients-by-coast-down-testing.html).

Para comparar a planta da foto, o script usa tambem as entradas conhecidas
das fases com motor: `tau*dq/dt+q=K*u_motor`, `dv/dt=q-perdas(v)`.
A primeira repeticao ajusta as perdas; a segunda escolhe sua complexidade;
o ajuste final usa ambas. K e tau tambem sao ajustados somente nas duas
primeiras; a terceira fica reservada para validacao. O modelo dinamico e avaliado acima de 0.45 m/s, incluindo
a tolerancia do ensaio de 0.5 m/s; nao descreve partida/aderencia em zero,
re ou rampas. Com somente duas repeticoes, a selecao usa BIC no treino.
Identificacao de uma transferencia exige definir sua entrada e sua saida,
conforme a [documentacao de identificacao de sistemas](https://www.mathworks.com/help/ident/gs/about-system-identification.html).

A linearizacao em velocidade positiva v0 e:

```text
G_u(s) = 1.26*K / [(tau*s+1)*(s+c1+2*c2*v0)]
```

O fator 1.26 vem de dois motores com o ganho de torque 0.63 utilizado em
`Car.set_u()`. Se as perdas forem constantes, a dinamica incremental e um
integrador com o eventual atraso do motor. Para obter numerador unitario
como na foto, e preciso normalizar a entrada pelo ganho identificado.
Nao se deve obter K ou tau somente da banguela, nem assumir que tau=0
fisicamente porque o atraso nao ficou resolvido na amostragem.

`identificacao.json` registra coeficientes, grade de K/tau, erros por velocidade
e a compatibilidade com a forma da foto. `validacao.png` compara os dados das
repeticoes reservadas com as previsoes. O criterio de comparacao padrao e
RMSE <= 0.05 m/s por velocidade; ele nao e uma garantia de desempenho de
um controlador futuro.
