# Stanley no veiculo real, usando o detector HSV existente

A missao reutiliza `tools/yellow_path.py`, o mesmo `YellowPathDetector` da
interface `tools/detectar_linha_amarela_hsv.py`: HSV forte/tolerante, CLAHE,
selecao de uma faixa continua, ajuste cubico, preenchimento de falhas curtas,
conexoes de pontilhado e suavizacao temporal com evidencia atual. O detector original e seus ajustes
foram preservados. A GUI de deteccao continua podendo ser executada sozinha.

`main.py` recebe o centro da faixa em pixels, converte para coordenadas do
chao em metros e calcula os erros lateral e angular no eixo dianteiro. A lei
Stanley calcula o estercamento, com saturacao e limite de taxa. Nao usa pose,
camera virtual ou localizacao ideal do CoppeliaSim.

## Calibrar antes de movimentar

A calibracao medida esta em `calibracao/camera_lateral.json` (12 pontos,
640x480). Ela corresponde a montagem usada na medicao. O arquivo
`calibracao/camera_lateral.example.json` e apenas um modelo vazio. A missao
valida o arquivo escolhido antes de criar `Car` ou abrir os atuadores.

1. Deixe a camera fixa, com o mesmo enquadramento e resolucao usados na missao.
2. Com o carro parado, marque pelo menos quatro pontos no chao, distribuidos
   em duas distancias e nos dois lados. Meca a partir do CENTRO DO EIXO
   DIANTEIRO: x positivo para a frente, y positivo para a esquerda, em metros.
   Os pontos nao podem ficar todos sobre uma unica reta.
3. Na Raspberry, abra uma foto da camera ou capture uma imagem:

```sh
python3 veiculo_real/tools/calibrar_camera_lateral.py --camera 0
# Alternativa com uma foto ja capturada pela mesma camera:
python3 veiculo_real/tools/calibrar_camera_lateral.py --image foto_da_camera.jpg
```

4. Clique nos pontos medidos, informe `frente esquerda` em metros e salve.
   A ferramenta gera `veiculo_real/calibracao/camera_lateral.json`. Ela abre
   somente a camera, sem criar `Car`, servos ou motores.
5. Se ja tiver os pares medidos, use `--points pares.json`, sem janela. O JSON
   deve conter `image_size: [largura, altura]`, `image_points_px: [[u,v],...]`
   e `ground_points_m: [[frente,esquerda],...]`, na mesma ordem.

A homografia assume chao plano. Mudancas na altura, inclinacao, pan da camera
ou resolucao exigem nova calibracao. Distorcao de lente nao e estimada por
esta ferramenta; as correspondencias devem representar a camera utilizada.

## Executar a missao

Na raiz do projeto, no ambiente Python ja preparado para o hardware:

```sh
python3 veiculo_real/main.py
# Parametros opcionais:
python3 veiculo_real/main.py --calibration caminho/camera.json --speed 0.5 --display
```

A referencia padrao e 0.7 m/s (`MAIN_VEL`); `--fast` usa 1.5 m/s
(`FAST_VEL`) e `--speed` aceita qualquer valor. Na GUI de um carro, o seletor
"Velocidade" troca o `--speed` do comando. O piso de velocidade em curvas e
0.15 m/s. O maximo de estercamento e 20 graus, com taxa de 120 graus/s.

O Stanley soma o esterco de regime `atan(L*kappa)` (`stanley_curvature_ff`)
e os erros da imagem sao levados ao instante do comando por odometria
(`vision_latency_compensation`, ate 0.25 s). O carro nao freia: so corta o
acelerador e desacelera ~0.6 m/s^2 (ensaio de banguela). A 1.5 m/s isso exige
~1.6 m para chegar a velocidade de curva, mais do que a camera enxerga com
`roi_top=50` (~0.85 m); curvas fechadas em velocidade alta dependem da direcao.

Perdendo a faixa, o carro segue a ultima trajetoria vista por ate
`line_loss_tolerance` (0.5 s) e so entao para.

### Ver a camera no notebook

Com o notebook no hotspot do carro, rode a missao com `--stream` (porta 8080)
e abra `http://<ip-do-carro>:8080/` no navegador, ou use o botao "Ver camera"
da GUI. A imagem mostra a faixa detectada, o status do controle, v, vref e o
esterco. So comprime quadros enquanto alguem assiste.
Ganhos, sinais e limites ainda precisam de ajuste e verificacao no carro
fisico. O entre-eixos do `Car` real continua 0.36 m; nao foi substituido pelo
valor de 0.302 m do simulador. `steering_sign=+1` segue a convencao do codigo
real (positivo para esquerda) e pode ser invertido se a montagem do servo
exigir. A calibracao de trim existente nos servos e preservada.

## Firmware do encoder e transferencia pela GUI

O `main.py`, o construtor `Encoder()` e o teste individual usam `rc` por
padrao, como a `upstream/rc-input` e o firmware deste repositorio. Nesse
modo sao obrigatorios os cinco campos `RPM,RC_DIR,RC_ACEL,SEL_DIR,SEL_TRACAO`;
as chaves publicadas pelo Arduino escolhem RC (0) ou AUTO (1).

Para firmware antigo que comprovadamente envia apenas um numero por linha
(por exemplo, `0.00`), selecione `--encoder-protocol rpm` na missao e
`--protocol rpm` no teste individual. Esse modo usa controle automatico de
direcao e tracao; nao recebe chaves ou comandos do radio. Nao existe troca
automatica de protocolo ao receber uma linha incompleta. Selecionar `rpm`
com firmware RC descarta todos os pacotes e mantem os comandos em neutro.

Na interface `fva_gui_multiplataforma.py`, envie novamente o conteudo de
`veiculo_real` para `/home/alunos/Desktop/fva`. A transferencia copia o
arquivo `.ino`, mas nao grava o Arduino. Com a missao encerrada, na aba
Executar Comandos, teste o protocolo RC sem comandar motores:

```sh
python3 -u /home/alunos/Desktop/fva/fva_car/encoder.py --protocol rc --seconds 5
```

Depois de aparecer `OK`, o ensaio de linha em baixa velocidade usa:

```sh
python3 -u /home/alunos/Desktop/fva/main.py --speed 0.3
```

Sem amostra atual do encoder por mais de 0.3 s, ou com pulso ausente em um
eixo selecionado para RC, o comando vai para neutro e o integral do PI e
zerado. Pulso zero nao e mais convertido em pedido de re. O CSV registra
`encoder_valid`; a missao informa mudancas de validade no terminal.

Em `main.py:parameters`, `path_settings` recebe os mesmos campos de
`tools/yellow_path.PathSettings`, por exemplo:

```python
"path_settings": {
    "h_min": 18, "h_max": 40, "s_min": 25, "v_min": 45,
    "clahe": True, "max_gap": 36,
    "dashed": True, "dash_gap": 120,
},
"path_anchor": 0.5,
```

H usa a escala OpenCV 0..179; S/V usam 0..255. Os ajustes feitos nos sliders
da GUI nao sao transferidos automaticamente: copie-os para `path_settings`.
A missao trabalha na resolucao da calibracao, sem o redimensionamento de
visualizacao da GUI. `path_anchor` escolhe a preferencia horizontal da faixa.

O modo `dashed` usa as conexoes entre trechos introduzidas no commit
`afabe3e` (`fix: yellow`). `dash_gap` e o intervalo maximo na escala de uma
imagem com 480 pixels de altura; aceita inteiros de 0 a 480. Desligar `dashed`
mantem apenas a recuperacao de falhas curtas por `max_gap`. Os dois ajustes
passam pela mesma validacao antes de inicializar o hardware.

O Stanley respeita a validade do `YellowPathDetector`, incluindo pontilhados
aceitos com menos de 50% do corredor pintado. O detector continua exigindo
evidencia atual e tangentes compativeis dos dois lados da conexao. A area
azul aparece tambem na visualizacao da missao.

O CSV inclui `line_color_support`, fracao do corredor com evidencia de cor,
e `line_dash_fraction`, fracao interpolada entre trechos. Ambos usam 0..1 e
recebem NaN quando a observacao nao e valida/atual. Suporte nao e probabilidade
de acerto; uma conexao azul tambem pode preencher um buraco em faixa continua.
Esses valores servem para diagnostico e nao alteram o ganho Stanley.

A velocidade e reduzida pela curvatura e pelos erros. Perda de deteccao,
frame mais antigo que `vision_timeout`, falha na thread ou emergencia cortam throttle
e zeram o integral do PI. A parada por obstaculo ja existente permanece
travada ate o fim da missao. Cortar throttle coloca o ESC em neutro; isso
nao equivale a uma frenagem ativa e o carrinho ainda percorre uma distancia.
A atuacao fica na thread principal; a visao apenas publica observacoes.

## Carro parado apesar de a ferramenta HSV detectar a faixa

A captura da missao roda em uma thread separada do processamento. Uma fila
de tamanho um substitui frames antigos pelo mais recente, sem acumular
imagens enquanto a Raspberry processa. A camera solicita buffer de um frame
(dependendo do suporte do driver) e nao forca foco, exposicao manual ou codec
diferentes da ferramenta `detectar_linha_amarela_hsv.py`.

O terminal imprime `CONTROLE:` uma vez por segundo com o motivo atual,
idade da imagem, duracao da captura, duracao do processamento, referencia
de velocidade e throttle. Por exemplo, `Imagem atrasada: 1.420 s; limite
1.000 s` indica que a observacao expirou. O horario continua sendo o inicio
da aquisicao, nunca o fim do processamento. O limite atual e 1.0 s, definido
em `vision_timeout`; ele nao representa a idade desejada para o controle.
Se o processamento sozinho ultrapassar esse limite, descartar frames na
fila nao basta: sera necessario medir e otimizar o processamento no hardware.

A missao nao desenha a visualizacao em cada frame: produz o destaque apenas
para `--display` (uma vez por segundo) ou ao salvar o diagnostico no fim.
O detector compartilhado descarta candidatos geometricamente impossiveis
em lote e valida primeiro os caminhos de maior score, conservando os
criterios de aceitacao, desempate e reconstrucao de pontilhados. O terminal
separa `deteccao` (HSV e busca da faixa) de `geometria` (projecao em metros e
ajuste da curva). Os mesmos tempos sao gravados como `vision_detection_s`
e `vision_geometry_s` no CSV e `detection_s` e `geometry_s` no JSON.

Para reduzir ainda mais o custo, use `--vision-width 320` ou `480`:

```sh
python3 -u main.py --speed 0.3 --vision-width 320
```

A camera continua capturando na resolucao calibrada (por exemplo, 640x480).
Somente o processamento usa uma imagem menor; os centros de pixel voltam
para a grade original antes da homografia. A calibracao nao precisa ser
refeita por causa dessa opcao. O padrao `--vision-width 0` preserva a grade
original. Menos pixels podem perder tracos muito finos: compare a faixa
desenhada e os erros metricos nos logs. O terminal mostra `grade=320x240`
para confirmar a opcao recebida; o JSON registra `processing_size`.
`visao=...fps` no terminal conta observacoes novas publicadas no intervalo,
incluindo capturas sem faixa; nao e a frequencia de 50 Hz do controle.
`vision_frame_id` no CSV identifica quais passos reutilizaram a mesma imagem.

## Resposta de direcao e ensaio em reta

O denominador do termo lateral do Stanley usa por padrao `abs(vref) + 0.3`.
O toggle `stanley_use_vref` em `main.py:parameters` permite comparar:

```python
"stanley_use_vref": True,   # referencia de velocidade deste passo
# "stanley_use_vref": False,  # velocidade medida e filtrada do encoder
```

A referencia e atualizada antes do calculo do Stanley, incluindo a reducao
por curvatura/erro e a selecao de tracao RC/AUTO. Nao se usa a referencia
zerada do passo anterior ao iniciar ou retomar o seguimento. O termo de
suavizacao, a saturacao e o limite de taxa permanecem os mesmos.
O CSV registra `stanley_speed_m_s` (entrada efetiva no denominador) e
`stanley_use_vref` (1 para referencia, 0 para medida); o modo tambem aparece
no terminal ao iniciar a missao.

O Stanley limita a taxa a 60 graus/s. A missao usa adicionalmente uma media
movel de apenas tres amostras no servo, em vez das dez anteriores. No ensaio
analisado, o atraso de dez amostras estimado pelos periodos registrados foi
de aproximadamente 188 ms (mediana), alem da idade da imagem. O log antigo
mostrava somente o comando ANTES dessa media. O novo `steer_filtered_rad`
mostra o comando apos o filtro de software; nao mede o angulo fisico da roda.

Para comparar os filtros explicitamente:

```sh
python3 -u main.py --vision-width 320 --speed 0.3 --steering-filter 3
python3 -u main.py --vision-width 320 --speed 0.3 --steering-filter 1
```

O valor 1 elimina a media movel; o limite de taxa do Stanley continua ativo.
O sinal da direcao nao foi invertido. Perda da faixa continua cortando o
throttle, zerando a direcao e reiniciando o Stanley, sem seguir dados ausentes.

Na missao, `imu_telemetry=False` evita duas leituras I2C por passo cujos ganhos
de fusao ja eram zero. O modelo usado pelo controle permanece o mesmo.
As colunas `a_x` e `w_imu` ficam NaN (nao medidas), em vez de inventar zeros.
Use `--imu-telemetry` para reativar essas leituras. Outros programas `Car`
mantem o comportamento anterior quando nao especificam essa opcao.
As medias moveis usam soma incremental, evitando criar vetores NumPy em
cada passo, e continuam recuperando-se quando NaN/Inf sai da janela.

O detector tambem calcula em lote os scores de conexoes curtas e as bandas
de candidatos, usa somas prefixadas para suporte de cor e resolve as pequenas
tangentes diretamente, sem uma SVD por ajuste. Os criterios de rejeicao,
validacao do pontilhado e o horario original da imagem continuam em vigor.

Para comparar as tres larguras usando uma foto coletada do proprio carro,
sem abrir camera ou motores, execute na pasta remota do projeto:

```sh
python3 tools/benchmark_visao.py --image logs/AAAAMMDD_HHMMSS/vision_camera.jpg
```

O resultado inclui mediana, percentil 95, tempos de deteccao/geometria e
quantidade de frames validos. `processing_hz` mede apenas processamento da
foto; nao representa o framerate garantido com sensores e controle rodando.

`Buscando faixa` indica rejeicao pelo detector HSV; `Faixa sem cobertura
metrica suficiente junto ao eixo dianteiro` indica faixa visual encontrada,
mas geometria rejeitada antes do Stanley. `Faixa valida; controle liberado`
indica que o seguidor autorizou comandos; as chaves RC/AUTO continuam
selecionando a origem de cada eixo. Uma parada por obstaculo continua travada
ate o fim do ensaio.

Ao encerrar uma missao com `save=True`, depois de neutralizar os motores,
a pasta `logs/AAAAMMDD_HHMMSS` tambem recebe o ultimo frame usado no controle:

- `vision_camera.jpg`: imagem original BGR gravada como JPEG.
- `vision_detection.jpg`: faixa e conexoes desenhadas sobre a imagem.
- `vision_status.json`: motivo de parada, tempos, resolucao, erros metricos
  e estados do encoder e das chaves.

Esses arquivos sao baixados junto com `car.csv` pelo botao `Coletar dados`
da GUI. O CSV inclui `vision_capture_s` e `vision_processing_s`, alem de
`vision_age_s`. Nao e necessario abrir uma janela grafica na Raspberry.

## Arquivos principais e verificacao

- `fva_car/lateral.py`: adaptador do detector, homografia e Stanley real.
- `main.py`: percepcao em thread, controle e prioridade de parada.
- `tools/calibrar_camera_lateral.py`: medicao pixel/chao.
- `fva_car/car.py`: `log()` e CSV com as colunas laterais, preenchendo NaN nas
  amostras anteriores. O formato DATA da GUI remota permanece com 10 campos.

```sh
MPLBACKEND=Agg python3 -m unittest discover -s veiculo_real/tests -v
MPLBACKEND=Agg python3 -m unittest discover -s veiculo_real/tools/tests -p test_yellow_path.py -v
```

Os testes usam imagens e atuadores simulados, sem conectar GPIO/I2C, camera
ou motores. A transferencia de arquivos para a Raspberry e a validacao no
hardware nao foram executadas nesta integracao.
