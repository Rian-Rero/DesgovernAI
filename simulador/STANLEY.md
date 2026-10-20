# Seguimento da linha amarela: Stanley + PI

## Objetivo e arquitetura

Primeiro validar o seguimento da linha amarela pela **camera do CoppeliaSim**
na cena `simulador_cones.ttt`. A linha e a trajetoria central desejada do carro,
nao uma borda de faixa a ser seguida com deslocamento lateral.

O Stanley controla o esterçamento. O controle longitudinal continua usando
o PI identificado em `fva_car/speed_controller.py`. A velocidade de referencia
e limitada nas curvas por `v <= sqrt(a_lateral_max / |curvatura|)`.

```text
Camera RGB -> mascara HSV amarela -> projecao metrica no chao
           -> ajuste local da linha -> erro lateral + erro de orientacao
           -> Stanley -> adaptacao do sinal -> direcao

Curvatura estimada -> referencia de velocidade -> PI existente -> motores
Velocidade medida ---------------------------------^
```

O nucleo em `autonomy/` nao depende de CoppeliaSim, GPIO, servos ou camera USB.
O adaptador em `simulador/seguir_linha.py` calibra a camera virtual e aplica os
comandos. O `main.py` original continua executando seu ensaio longitudinal;
o novo ensaio e iniciado explicitamente por `seguir_linha.py`.

## Lei de controle e convencoes

Baseado no controlador apresentado na aula 08, paginas 60-61:

```text
delta = saturar(psi + atan2(k * e, v_soft + |v|), -delta_max, +delta_max)
```

`e` e a distancia normal assinada entre o centro do eixo dianteiro e a linha,
em metros. `psi` e o angulo entre a tangente da linha e o eixo longitudinal
do carro, em radianos. `v` e a velocidade medida em m/s.

Convencao do nucleo: x para a frente, y para a esquerda, erro positivo quando
a linha esta a esquerda, esterçamento positivo para a esquerda.

Na cena fornecida, foi medido que **`Car.set_steer` positivo vira para a direita**.
Por isso o adaptador aplica `-delta`. Essa inversao nao faz parte da lei
Stanley e precisa ser verificada novamente no servo real.

Valores atuais do ensaio: `k = 5.0 1/s`, `v_soft = 0.2 m/s`, limite de 20 graus e
taxa maxima de 90 graus/s. A regularizacao em baixa velocidade, o limite
de taxa e a variacao de velocidade sao extensoes praticas; nao se deve
transferir automaticamente a prova de estabilidade do caso ideal da aula
para um sistema com visao, atrasos e saturacoes.

O ganho foi ajustado a partir dos logs e de ensaios com k=1.5, 2.5, 3.5,
5 e 7. O comparativo e as condicoes de validacao estao em
[ensaios/SINTONIA_STANLEY.md](ensaios/SINTONIA_STANLEY.md). A configuracao
e as parcelas do controlador sao registradas em cada novo ensaio.

## Visao e calibracao

1. Capturar RGB da camera virtual; `Car.get_image` ja inverte verticalmente
   o buffer do CoppeliaSim. Nao inverter novamente nem tratar RGB como BGR.
2. Segmentar amarelo no HSV do OpenCV: H=17..38, S=90..255, V=65..255.
3. Projetar a mascara no plano do chao, em metros. A origem fica na projecao
   do centro do eixo dianteiro. Isso evita misturar pixels com velocidade em m/s.
4. Percorrer a linha do trecho proximo para o distante e ajustar sua geometria
   **local**. Usar reta quando os residuos estiverem abaixo de 5 mm; nos outros
   casos, ajustar `y(x)` e avaliar um arco circular. O arco e usado quando sua
   qualidade permite extrapolar melhor a tangente numa curva fechada.
   O ajuste usa aproximadamente 0.8 m de extensao a partir da primeira parte
   visivel, no maximo ate x=1.4 m. Ajustar toda a imagem com uma unica parabola
   pode misturar curvas opostas.
5. Calcular o ponto da curva mais proximo do eixo dianteiro, sua normal e
   tangente. Rejeitar linhas com pouca extensao, ajuste ruim ou geometria extrema.

A camera virtual e inclinada a **30 graus para baixo durante o ensaio**,
com `--camera-pitch 10` disponivel para reproduzir a montagem original. A
montagem original de cerca de 10 graus deixa uma regiao proxima do carro
fora da imagem e dificultou a estimativa nas curvas fechadas. O script restaura
a montagem ao encerrar e nao grava alteracoes no arquivo da cena.

A calibracao automatica ocorre depois de 0.5 s com motores livres, para
assentar a suspensao. Ela le a resolucao, abertura, pose da camera e posicoes
das juntas dianteiras. Depois da calibracao, a realimentacao lateral usa somente
a imagem, sem consultar um mapa, posicoes de waypoints ou a linha na cena.

Hipoteses: pista plana, camera rigida e sem distorcao. O plano padrao e
`z=0.00205 m`, a superficie de `Pista` na cena de cones. A cena de rampa nao
esta validada: uma homografia fixa deixa de ser correta quando a camera
e o piso mudam de inclinacao relativa.

## Como executar

Na raiz do repositorio, instalar no ambiente virtual:

```bash
venv/bin/python -m pip install -r simulador/requirements.txt
```

Abrir `simulador/coppeliasim/simulador_cones.ttt` no CoppeliaSim e deixar a
simulacao parada. Comecar com:

```bash
venv/bin/python simulador/seguir_linha.py --vref 1.5 --ts 60
```

Para ensaios longos com logs e relatorio, sem plot ao vivo:

```bash
venv/bin/python simulador/seguir_linha.py --vref 1.5 --ts 360 --sem-grafico --sem-tempo-real
```

Opcoes: `--ganho` ajusta k, `--port` escolhe a porta ZMQ, `--camera-pitch`
ajusta a inclinacao da camera e `--ground-z` informa a altura do piso.
A simulacao continua renderizando a camera mesmo
com `--sem-grafico`: essa opcao so desativa a telemetria do Matplotlib.

Neste computador, o backend Wayland produziu uma imagem uniforme por falha
na criacao do contexto OpenGL. Iniciar o CoppeliaSim com
`QT_QPA_PLATFORM=xcb` resolveu a captura. Exemplo de instancia separada:

```bash
QT_QPA_PLATFORM=xcb /opt/CoppeliaSim/coppeliaSim.sh -h \
  -GzmqRemoteApi.rpcPort=23010 \
  "$PWD/simulador/coppeliasim/simulador_cones.ttt"
# Em outro terminal:
venv/bin/python simulador/seguir_linha.py --port 23010 --vref 1.5 --ts 60
```

Usar `-h`, que apenas oculta a janela. O modo `-H` do CoppeliaSim nao oferece
a mesma renderizacao OpenGL da camera.

## Perda da linha e diagnosticos

O carro aguarda tres quadros validos antes de partir. Uma falha curta de
visao conserva o ultimo esterçamento e reduz a referencia. Sem observacao
valida por mais de 0.15 s, zera a referencia, reseta o PI e freia; ja parado,
libera o torque e centraliza a direcao. A retomada requer tres quadros validos.
Imagem uniforme, linha branca e deteccoes de baixa confianca nao autorizam partida.

Cada ensaio salva em `logs/<data>/`:

- `car.csv`: sensores, comandos, erro lateral, erro angular, curvatura,
  confianca, idade da observacao e estado de parada;
- `camera_calibration.json`: homografia e convencoes da camera virtual;
- `controller_config.json`: ganho aplicado, referencia, limites e periodo;
- `camera_overlay.png`, `yellow_mask.png`, `bird_mask.png`: ultimo quadro,
  deteccao e projecao;
- `stanley_report.png` e `stanley_metrics.json`: trajetoria medida, velocidade,
  esterçamento e estatisticas depois dos primeiros 5 s.

O RMS lateral desses relatorios e **estimado pela propria visao**. Ele nao
mede sozinho o erro verdadeiro: uma calibracao com vies pode produzir uma
estimativa pequena. Para validacao final, comparar tambem a trajetoria do
eixo dianteiro com uma referencia independente, usada apenas na avaliacao.

Foi incluido tambem um avaliador independente da camera de controle:

```bash
venv/bin/python simulador/avaliar_stanley.py logs/<pasta-do-ensaio> --port 23000
```

Ele le a textura e as coordenadas UV de `/Pista`, extrai a linha central e
compara as posicoes registradas no log com essa referencia. Gera
`reference_metrics.json` e `reference_trajectory.png` (azul: trajetoria medida,
verde: inicio, vermelho: fim). O ponto avaliado e o **centro do corpo do carro**,
nao o eixo dianteiro; os dois erros diferem durante as curvas. A referencia
tem resolucao finita de textura, informada no JSON, e e usada somente offline.
O CoppeliaSim deve estar aberto com a mesma pista usada no ensaio.

Testes de geometria, sinais, convergencia do modelo bicicleta e perda de linha:

```bash
MPLBACKEND=Agg venv/bin/python -m unittest discover -s simulador/tests -v
```

## Passagem para o carrinho real

Esta etapa vem depois da validacao de circuito e repetibilidade no simulador.
O ensaio novo nao altera o acionamento do carrinho real.

1. **Visao offline.** Gravar quadros da webcam parada na pista real. A API
   `veiculo_real/fva_car/camera.py` entrega BGR: usar
   `YellowLineDetector(..., color_order="BGR")`. Ajustar HSV para iluminacao,
   sombras e exposicao reais. Verificar a mascara antes de mover o carro.
2. **Calibracao fisica.** Medir distancia entre eixos, posicao da camera,
   altura/inclinacao e centro do eixo dianteiro. Calibrar intrinsecos e distorcao
   da lente. Obter homografia com pelo menos quatro pontos nao colineares do
   piso, com coordenadas medidas em metros nesse referencial. Corrigir a
   distorcao antes de detectar a linha. Nao reutilizar a homografia do simulador.
3. **Direcao.** Com tracao desabilitada, verificar o sinal do servo, trim,
   ganho, limites mecanicos e atraso. O atuador real ja aplica media movel
   de dez amostras; esse atraso precisa entrar na sintonia lateral.
4. **Longitudinal.** Usar a velocidade do encoder e o controlador especifico
   do carro real. O PI do simulador atua em torque com uma planta diferente;
   os ganhos e a conversao de comando nao devem ser copiados sem identificacao.
5. **Integracao.** Publicar observacao, confianca e timestamp da camera para
   o controle; rejeitar quadros atrasados. Manter a parada por obstaculo do
   ultrassom com prioridade sobre o seguimento, e prever parada manual.
6. **Ensaios graduais.** Validar sinais e erros com o carro parado, depois
   seguir em baixa velocidade. Aumentar velocidade somente apos avaliar
   erros, saturacao da direcao, atrasos e distancia de parada em repeticoes.

Critérios para avancar: acompanhar o circuito inteiro em repeticoes, nao
trocar de faixa nas curvas, permanecer dentro da margem fisica disponivel,
parar ao perder a linha e medir a distancia de parada. Os limites numericos
de erro aceitavel devem vir da largura da linha/pista e do carrinho real.

## Resultado inicial de validacao - 2026-10-20

Ensaio final do trecho esquerdo, iniciado antes do canto da reta diagonal,
com camera a 30 graus, k=1.5 e referencia de 0.8 m/s:

| Medida | Resultado |
| --- | ---: |
| Duracao | 130.05 s |
| Distancia percorrida | 102.64 m |
| Deteccoes validas depois de 5 s | 100% |
| Paradas por perda de linha depois de 5 s | 0 |
| RMS do erro lateral estimado no eixo dianteiro | 4.26 cm |
| RMS independente, centro do corpo versus linha da textura | 2.02 cm |
| Percentil 95 do erro independente | 4.07 cm |
| Pico do erro independente | 10.46 cm |
| Resolucao espacial da referencia de textura | 1.8 cm/pixel |

Registro local: `logs/20261020_194350_075542/`. O ensaio anterior de curvas
fechadas percorreu 26.29 m a 0.6 m/s, tambem sem perdas depois da partida.
Esses resultados iniciais cobrem trechos do circuito em ensaios distintos.
Naquela etapa, ainda nao havia sido comprovada uma volta completa continua;
a tentativa longa anterior motivou o ajuste de visibilidade no canto da diagonal.

Para repetir as condicoes iniciais do ensaio final, com simulacao parada:

```bash
venv/bin/python simulador/seguir_linha.py --ganho 1.5 --vref 0.8 --ts 130 \
  --initial-pose -13.5179196 -0.3151322 -141.016836 \
  --sem-grafico --sem-tempo-real
```

`--initial-pose X Y YAW_DEG` usa coordenadas mundiais e orientacao do eixo
longitudinal do carro. A altura da cena e preservada e a pose original e
restaurada ao encerrar. Para a verificacao de volta completa, omitir essa opcao
e testar a configuracao final durante pelo menos 360 s a 0.8 m/s.

Validacao posterior da sintonia k=5 em 2026-10-20: 260 s com referencia
de 1.5 m/s, 356.26 m percorridos e nenhuma perda depois da aquisicao. O
retorno pela reta inicial confirmou uma volta completa em 181.4 s. Consulte
o relatorio de sintonia para os erros e as condicoes dessa validacao.
