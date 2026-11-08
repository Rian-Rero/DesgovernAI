# Centro dos cones no carrinho real

O modo padrão `--navigation cones` usa somente cones. Não há alternância
para linha amarela. Se o detector perder a referência métrica de cones, o
controle segue a trajetória já identificada por até `cone_loss_frames`
quadros (padrão 6) e `cone_loss_tolerance` segundos (padrão 0,6), reduzindo
a vref em rampa até `curve_min_speed`. Se a perda continuar, neutraliza os
comandos. O modo de linha continua disponível apenas por seleção explícita com
`--navigation line`.

A GUI `tools/fva_gui_um_carro.py` já inicia nesse modo com `--stream 8080`.
O comando da GUI usa `--vision-width 0` para preservar a resolução calibrada
e os detalhes dos cones próximos ao horizonte.
O seletor mantém as velocidades 0,7, 1,0 e 1,5 m/s; o valor inicial é 0,7 m/s.
O exemplo abaixo usa 0,3 m/s pela linha de comando.

O detector de cones usa a câmera BGR e a calibração do piso para estimar as
posições das bases dos cones. Vermelhos e verdes podem trocar de lado:
com `auto_sides` (padrão), a geometria decide qual cor fica à esquerda,
comparando a posição lateral das duas fileiras no piso. Para trocar um lado
já decidido, são precisos `side_flip_frames` quadros seguidos (padrão 3) com
as duas cores mostrando o contrário. Um quadro ruidoso não inverte as bordas,
e o quadro não é mais reprocessado com as cores trocadas.
Não usa as coordenadas do CoppeliaSim nem exige linha amarela.

Execute no computador embarcado, na raiz do projeto:

```sh
python3 veiculo_real/main.py --navigation cones --speed 0.3 --stream 8080
```

A transmissão mostra as bases vermelhas/verdes e o centro calculado em roxo (tracejado quando estimado).
`--display` exibe o resultado localmente. O modo padrão é cones; `--navigation line` seleciona somente a linha amarela.

## Geometria e controle

`fva_car/cones.py` agora adapta o mesmo `ConePathDetector` de
`tools/cone_path.py` usado por `detectar_linha_amarela_hsv.py`. O ROI padrão
é 0 para cones e linha amarela: nenhuma faixa do topo é recortada.

A sequência compartilhada aplica mediana 3×3, segmentação HSV, fechamento
morfológico e validação do formato triangular. Seleciona cadeias contínuas de
cones e projeta as bases pela homografia de `camera_lateral.json`, com origem
no eixo dianteiro, +x para frente e +y à esquerda. A câmera deve permanecer
na posição usada na calibração. Cada borda observada precisa de pelo menos
três cones e da extensão mínima configurada.

O centro é calculado entre curvas por seções normais no piso, com validação de
largura, margem e raio mínimo.

**Dois caminhos visíveis.** Cada cor gera até 3 cadeias candidatas: a melhor,
uma disjunta (outra fileira da mesma cor) e variantes que partem dos mesmos
cones (ramos de uma bifurcação). Até 3 combinações esquerda × direita passam
pela geometria métrica. Vence a de menor custo, e o termo de maior peso é a
**continuidade com a trajetória anterior**, levada ao referencial atual pelo
ego-movimento, enquanto ela tiver menos de `path_prior_ms` (padrão 1000 ms).
Sem histórico, vence o corredor que contém o carro. Quando dois corredores
válidos divergem mais de 0,3 m, o quadro é marcado como bifurcação, e as
alternativas aparecem em cinza no vídeo.

**Reta ou curva.** O centro recebe um perfil de curvatura a cada 0,25 m de arco.
O trecho próximo (0,5 m) é classificado como `reta`, `curva_esq` ou `curva_dir`
(limiar 0,25 1/m, raio de 4 m), e `curve_ahead_m` indica a distância até a
próxima curva. O rastreamento e a memória são os mesmos da
interface: cones persistem por até 350 ms; uma borda visível pode usar a largura
aprendida por até 1500 ms, sujeita ao limite de confiança. `tracking=False`
desativa essa estimativa. Quando não resta uma referência de cones válida, o
Stanley entra em *hold*. Prevê a trajetória por odometria, sem esterço de busca,
e reduz a vref em rampa. Também descarta saltos impossíveis entre quadros
(`cone_max_lateral_jump_m` 0,25 m, `cone_max_heading_jump_rad` 0,35 rad).
Esgotada a tolerância, neutraliza os comandos. `--loss-frames N` ajusta a
tolerância. Com `--loss-frames 0`, para no primeiro quadro perdido. O adaptador calcula os erros do
Stanley junto ao eixo dianteiro a partir do centro métrico.

## Debug por quadro (leve para o Raspberry Pi 4)

Com `cone_debug` (padrão; `--no-cone-debug` desliga), cada ensaio em
`logs/<data>/` ganha:

- `cones_debug.jsonl`: um JSON por quadro de visão, com
  - `vision.detection`: bases aceitas por cor (px e m) e componentes
    rejeitados por motivo (`pequeno`, `proporcao`, `nao_triangular`,
    `cortado_borda`…);
  - `vision.tracking`: cones rastreados com idade, fonte do movimento e delta
    do carro;
  - `vision.sides`: lado decidido, evidência, placar e trocas;
  - `vision.chains` e `vision.corridors`: candidatos, custos, escolha, motivo
    e bifurcação;
  - `vision.trajectory`: reta/curva, raio, curva à frente, perfil de
    curvatura, larguras, centro e bordas no piso, erros e confiança;
  - `control`: hold, quadros perdidos, fator da vref, esterço e status.
- `cone_frames/`: JPGs de 320×240 **somente em eventos** (perda, reaquisição,
  parada, salto descartado, troca de lado, bifurcação), com 5 quadros antes e
  5 depois. O limite é `cone_max_event_frames` (300, cerca de 4 MB).
- `cones_summary.json`: quadros válidos, estimados, em hold e perdidos, maior
  perda, eventos, trechos, fps e tempo médio de detecção.

A gravação roda numa thread separada. Com a fila cheia, o registro é
descartado e contado, sem atrasar o controle. A GUI `fva_gui_um_carro.py` baixa
somente arquivos novos, mostra o gráfico "Cones" e o resumo. No notebook:

```sh
python3 veiculo_real/tools/analisar_cones.py experimentos/<carro>/<data>
python3 veiculo_real/tools/analisar_cones.py experimentos/<carro>/<data> --frame 123
python3 veiculo_real/tools/analisar_cones.py experimentos/<carro>/<data> --plot 100 160
```

O Stanley já existente recebe erro lateral, orientação e curvatura desse centro.
Mantém antecipação pela curvatura, redução de velocidade em curvas, compensação
de atraso, sinais/limites do servo e proteções de encoder, RC, emergência e
obstáculos. Quando só há uma borda confiável, o modo de cones estima o centro
pelo lado observado e pela largura aprendida ou nominal. Um cone isolado só
orienta a recuperação se persistir e estiver próximo e claramente de um lado.
Com o FOV atual, o primeiro par de cones só aparece a cerca de 1,3–2,3 m.
Se o centro começa além de 1,5 m e o trecho é reto (|κ| < 0,25 1/m), ele é
extrapolado até o eixo dianteiro, aceitando um início em até 2,6 m
(`CONE_FAR_START_M`), e a referência fica limitada a 0,35 m/s. Em curva
distante isso não é feito, porque a reta secante inventaria erro lateral.
A recuperação limita a referência a 0,25 m/s. Se perder a referência, o
Stanley segue brevemente a última trajetória pela odometria e então para.

## Calibração e ajustes

A calibração atual contém medidas a 0,30–0,60 m do eixo dianteiro. Ela não
comprova precisão em toda a área necessária para ver os cones. Confira o
campo de visão e calibre com pontos no piso cobrindo as distâncias e larguras
usadas. Se mudar a inclinação da câmera para enxergar mais longe ou as duas
bordas, refaça a calibração com `tools/calibrar_camera_lateral.py`.
A homografia vale para o contato no piso, não para o topo dos cones.

Em `main.py`, `cone_settings` permite ajustar:

- `roi_top`: porcentagem do topo ignorada (padrão 0).
- `green_h_min` / `green_h_max`: faixa de verde (padrão 38–90).
- `red_hue`: largura das faixas vermelhas nos extremos de H (padrão 18).
- `s_min` / `v_min`: mínimos de saturação e brilho do verde (65 e 45).
- `red_s_min` / `red_v_min`: mínimos próprios para vermelho escuro (45 e 35).
- `chroma_min` / `green_chroma_min`: croma Lab mínimo do vermelho e do verde (12 e 12).
  O verde-azulado escuro na sombra fica entre 4 e 12. Baixe só o do verde e
  confira os falsos positivos, porque a porta marrom tem croma entre 7 e 13.
- `partial_cones`: aceita o cone cortado só pela lateral da imagem quando a base
  e pelo menos 60% da altura estão visíveis (padrão True). O centro da base é
  estimado pela borda visível.
- `min_area`: área mínima na escala 640×480 (28 pixels).
- `min_span` / `max_gap`: extensão e continuidade das bordas na imagem.
- `min_width_cm` / `max_width_cm`: largura admissível (40–500 cm).
- `tracking`, `memory_ms`, `width_memory_ms`: rastreamento e memória.
- `auto_sides`, `side_flip_frames`: lados automáticos e histerese (True, 3).
- `path_prior_ms`: validade da trajetória anterior como prioridade (1000 ms).

Em `parameters` de `main.py`: `cone_loss_frames`, `cone_loss_tolerance`,
`cone_max_lateral_jump_m`, `cone_max_heading_jump_rad`, `cone_debug` e
`cone_max_event_frames`.

Os campos são os de `ConeSettings` da interface HSV. Com `auto_sides`,
`green_left` é só a preferência inicial; sem ele, é o lado fixo.

H usa 0–179; S e V usam 0–255. Ajuste a segmentação conforme a iluminação.
A detecção por cor pode confundir objetos vermelhos/verdes com cones; esta
implementação ainda precisa ser validada com imagens e ensaios do veículo real.
Cones cortados na base (borda inferior da imagem) são descartados. Os cortados só
pela lateral são aceitos com `partial_cones`. Cones com faixa branca devem manter
a parte colorida inferior visível.

## Classificador de cones (rede neural)

A cor gera os candidatos. Uma CNN pequena (~24 mil parâmetros, ~100 KB) olha
um recorte de 32×32 em volta de cada base e decide entre `cone_verde`,
`cone_vermelho` e `nao_cone`. Só os cones confirmados, com a mesma cor da
máscara, seguem para as bordas. A posição da base continua vindo da geometria.
Com o classificador, a cor pode ser mais larga (`LOOSE_COLOR`) e pegar o verde
escuro na sombra, porque a rede descarta porta, roupa e guarda-chuva. No Pi
roda só com o OpenCV (`cv2.dnn`), sem PyTorch. O recorte é o mesmo no treino e
no carro (`tools/cone_classifier.py`).

Fluxo (o que só a equipe pode fazer está em **negrito**):

1. **Gravar** quadros na pista e na luz da prova. As curvas e sombras são o
   mais importante.
   `python3 veiculo_real/main.py --record-frames 3 --speed 0.3`
   Também dá para dirigir pelo rádio no modo manual; a visão grava do mesmo jeito.
2. **Rotular** no notebook. O detector pré-marca os cones e você corrige com
   cliques. Comece com uns 200–300 quadros revisados.
   `python3 veiculo_real/tools/rotular_cones.py logs/<ensaio>`
3. Gerar os recortes (positivos dos rótulos, negativos dos candidatos sem
   rótulo):
   `python3 veiculo_real/tools/extrair_recortes_cones.py logs/<a> logs/<b> --out cones_ds.npz --preview amostra.jpg`
4. Treinar no notebook (`pip install torch --index-url https://download.pytorch.org/whl/cpu onnx`):
   `python3 veiculo_real/tools/treinar_classificador_cones.py cones_ds.npz --out veiculo_real/modelos/cones.onnx`
   A validação é separada por quadro. O script mostra a matriz de confusão e
   recarrega o modelo no OpenCV para confirmar que a saída bate com o PyTorch.
5. Medir antes e depois:
   `python3 veiculo_real/tools/benchmark_visao.py --cones --image logs/<ensaio>/vision_camera.jpg --classifier veiculo_real/modelos/cones.onnx`
   e `avaliar_cones.py` nos quadros rotulados que ficaram fora do treino.
6. Usar no carro: `"cone_classifier": "veiculo_real/modelos/cones.onnx"` em
   `main.py`, ou `--classifier ...` na linha de comando (`--classifier none`
   desliga).

O `.json` ao lado do `.onnx` guarda as classes, o recorte, os ajustes de cor
dos candidatos usados no treino (aplicados automaticamente, salvo se
`cone_settings` disser outra coisa) e a confiança mínima (`min_confidence`,
padrão 0,6). O debug por quadro registra os candidatos, os aceitos e o tempo
do classificador em `vision.detection.classifier`, e os descartes como
`classificador`. O modelo em `tools/tests/dados/` é **só de teste**: foi
treinado com pseudo-rótulos.

## Visão em processo separado

No carro real (`vision_process: True`, padrão), a câmera e o detector rodam
num processo próprio, que é outro interpretador Python em outro núcleo. O
processo principal fica com o laço de controle a 50 Hz, o log e o stream. Num
processo só, os dois disputavam o GIL. Num teste com o controle ocupando 60%
de um núcleo, a detecção inflava de ~10 para 35 ms em thread e ficava em
~11 ms em processo.

- A câmera abre no processo de visão **antes** dos motores. Se ela falhar, a
  missão para com `Camera/visao: ...` sem mover o carro.
- A odometria chega ao detector por um histórico de poses em memória
  compartilhada (`SharedPoseHistory`).
- Cada quadro volta ao controle pronto, com a imagem original para o debug e
  o stream. As máscaras não são enviadas: ~0,9 MB e ~0,3 ms por quadro.
- Se o processo de visão morrer, o controle neutraliza o carro e encerra,
  como antes acontecia com a thread.
- `vision_process: False` volta às threads no mesmo processo. Um carro injetado
  (testes e simulação) sempre usa threads.

## Câmera: exposição e balanço de branco

Com exposição automática, a câmera escurece ou clareia conforme o que aparece
na imagem. Num corredor escuro, isso também reduz o fps. Fixar a imagem deixa
as cores dos cones estáveis entre quadros. No Raspberry, com o carro parado na
pista:

```sh
sudo apt install v4l-utils   # uma vez
python3 veiculo_real/tools/ajustar_camera.py
```

A ferramenta testa várias exposições e mostra fps, brilho, área estourada e
saturação dos cones. Ela salva uma foto por ajuste em `ajuste_camera/` e imprime
a linha para colar em `main.py`, por exemplo
`"camera_settings": {"fourcc": "MJPG", "exposure": 150, "auto_wb": False}`.
Com `camera_settings` vazio, a câmera fica no modo automático, como antes.

Para separar o custo do detector da disputa com as demais threads, rode no Pi
`python3 veiculo_real/tools/benchmark_visao.py --cones --image logs/<ensaio>/vision_camera.jpg`
e compare com o `detection_s` do log.

Os logs acrescentam `cone_red_count`, `cone_green_count`, `corridor_width_m`,
`cone_green_left`, `cone_estimated`, `cone_branch`, `cone_segment` (0 reta,
+1 esquerda, −1 direita), `line_hold`, `line_lost_frames` e `hold_speed_factor`.
`line_valid`, `cross_track_m`, `heading_error_rad` e `curvature_m_inv` continuam
representando a referência usada pelo controlador, agora o centro do corredor.

A transmissão mostra a contagem de cones aceitos por cor, inclusive quando
a geometria do corredor é inválida. Regiões unidas por sobreposição podem
fornecer a base inferior visível; bases ocultas não são inventadas.
