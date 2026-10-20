# Seguidor da linha amarela com Stanley

Na raiz do projeto, com CoppeliaSim aberto e a cena carregada:

```sh
python3 simulador/main.py
```

O programa usa a camera RGB, segmenta amarelo, escolhe segmentos continuos
por linha da imagem e projeta os centros no plano do chao. Intrinsecos e pose
vem do sensor do simulador. Nao requer OpenCV. O centro entre as juntas de
direcao define o eixo dianteiro. A frente do objeto Car e seu eixo Z local,
nao o yaw convencional do quaternion.

## Modelo

Modelo bicicleta cinematico, eixo traseiro, L=0.302 m:

    x_dot = v*cos(psi)
    y_dot = v*sin(psi)
    psi_dot = v*tan(delta)/L

Para pequenos angulos, velocidade constante e caminho reto, com erros
carro menos caminho:

    e_y_dot = v0*e_psi
    e_psi_dot = v0*delta/L
    G_r_delta(s) = v0/L
    G_psi_delta(s) = v0/(L*s)
    G_y_delta(s) = v0^2/(L*s^2)

`Controlador/lateral.m` disponibiliza essas plantas e os polos da aproximacao
linear do Stanley. Requer Control System Toolbox. Nao altera a planta
longitudinal existente. Modelo baseado em geometria, ainda sem identificacao
experimental de servo, pneus, atraso ou escorregamento; nao foi extraido do
PDF da disciplina. Valido para avanco, baixa velocidade e pista plana.

## Controle

No Python, os sinais da percepcao sao linha menos carro: `cross_track_m`
positivo para linha a esquerda; `heading_error_rad` positivo para direcao
da linha a esquerda. O erro lateral e a distancia normal a reta local no
eixo dianteiro, em metros. Lei Stanley:

    delta = heading_error + atan2(k*cross_track, abs(v)+v_soft)

`delta` e geometrico, positivo para esquerda. As juntas da cena Car tem
sinal contrario: `set_steer(steering_sign*delta)`, com `steering_sign=-1`.
O log de 20/10/2026 confirmou que comando positivo reduz a direcao de avanco
(vira para direita). Esse adaptador mantem o Stanley e o modelo em convencao
geometrica. O CSV separa `steer_geometric_rad` e `steer_cmd_rad`, registra o
sinal e as contagens de pixels e pontos projetados para diagnosticar perda.

Padroes: k=1.5 1/s, v_soft=0.3 m/s, saturacao +/-20 graus e taxa maxima
60 graus/s. O PI longitudinal usa ate 1.5 m/s. O ajuste local usa pontos
proximos ao eixo dianteiro e rejeita cobertura insuficiente e residuo alto.
O ajuste compara uma parabola com um arco circular nas curvas fechadas,
selecionando o arco somente quando reduz o residuo geometrico. No arco,
a tangente e o erro sao calculados no ponto mais proximo ao eixo dianteiro.
Na parabola: a tangente em x=0 determina o erro angular e a
curvatura local e registrada em `curvature_m_inv`. Isso evita usar a secante
de uma curva como se fosse a direcao junto ao carro. A velocidade desejada
e reduzida pela curvatura (`sqrt(curve_lateral_accel/abs(curvatura))`) e pelos
erros lateral e angular, com piso `curve_min_speed` e teto
`velocity_reference`. Padroes: 0.6 m/s^2 e 0.5 m/s. O piso pode impedir o
cumprimento do limite nominal em curvas extremas. Esses parametros sao
iniciais e ainda precisam de validacao na cena. O Stanley continua sem
termo adicional de antecipacao por curvatura.
Ao perder a linha, comanda velocidade zero; conserva a ultima direcao por
ate 0.4 s enquanto freia, depois centraliza. Sem historico valido, centraliza
imediatamente. Volta a
seguir quando a deteccao valida retorna. A parada tem distancia de frenagem.

`main.py:parameters` permite mudar `stanley_gain`, `velocity_reference` e
`line_ground_z`. Este ultimo deve coincidir com a altura real da pintura
(padrao 0.005 m); em rampas a projecao em plano horizontal nao e valida.
O carro precisa iniciar com a linha visivel. A faixa amarela escolhida e a
mais proxima ao centro na base da imagem, continuada nas linhas seguintes;
cruzamentos e outras marcas amarelas exigem selecao de rota adicional.

## Telemetria e verificacao

O seguidor conserva por ate 3 s os pontos de linha medidos no mundo,
usando a pose do simulador. Assim, pontos vistos antes continuam disponiveis
perto do eixo dianteiro mesmo quando saem da base da imagem. A nuvem e
limitada a observacoes proximas e reamostrada em celulas de 4 cm. Sem novos
pontos suficientes por mais de 0.3 s, a memoria deixa de autorizar controle
e a parada e acionada. Isso usa localizacao ideal do simulador; no carro
real seria necessario estimar a pose e considerar sua incerteza.

Na transicao de linha valida para perdida, o programa salva na pasta do log
`line_loss_TEMPO.npz`, com ate 20 imagens anteriores, pixels detectados,
pontos projetados, memoria local, pose da camera e eixo dianteiro. Os campos
tem sufixo de indice (`image_0`, `points_0`, `time_0` etc.) e podem ser lidos
com `numpy.load`, sem pickle. Esses dados permitem reproduzir a percepcao;
o CSV sozinho nao permite confirmar qual trecho da pintura foi selecionado.

O painel mostra a camera no topo, ocupando as duas colunas. Abaixo ficam
erro lateral e velocidade medida com referencia; na ultima linha, PWM e
estercamento em graus (comando equivalente e angulos reais das duas rodas,
na convencao de sinal das juntas). Os quatro graficos compartilham o eixo
de tempo. O erro lateral tem referencia zero e escala
vertical simetrica automatica; a curva e interrompida quando a deteccao e
invalida. Valores positivos indicam linha a esquerda.

O grafico de velocidade distingue a referencia nominal constante (vermelho
tracejado) da referencia ajustada por curvas/erro (laranja pontilhado),
efetivamente enviada ao PI. Pequenas variacoes da referencia ajustada
tambem refletem as variacoes da estimativa visual.

CSV inclui validade da linha, erros metricos e angulares, comando e angulos
reais das juntas, guinada bruta e direcao real do eixo frontal. As colunas
de comando lateral correspondem ao intervalo [t, t_proximo); a guinada
medida em t resulta dos comandos anteriores. `th` legado permanece intacto.

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/fva-mpl python3 -m unittest discover -s simulador/tests -v
```

Testes cobrem convergencia no modelo bicicleta, sinais, saturacao, limite
de taxa, projecao metrica, segmentacao e parada na perda da linha. Aferir na
cena real o sinal do esterçamento, altura da pintura, visibilidade e ganhos
antes de considerar o desempenho validado no CoppeliaSim.

Referencia da geometria do sensor:
https://manual.coppeliarobotics.com/en/visionSensorPropertiesDialog.htm
https://forum.coppeliarobotics.com/viewtopic.php?t=6333
