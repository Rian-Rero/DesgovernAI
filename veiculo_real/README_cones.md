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
tolerância. Com `--loss-frames 0`, para no primeiro quadro perdido. Sem o estimador
(`cone_estimator: False`), o adaptador calcula os erros do Stanley junto ao eixo
dianteiro a partir do centro métrico de cada quadro.

## Estimador do corredor (padrão)

A câmera vê os dois lados juntos só entre ~1,9 e 3 m. Ela está apontada uns 8°
para a direita, então o cone esquerdo entra bem depois do direito. E mede mal a
distância: 1 px ≈ 5 cm a 2,5 m, e 1° de arfagem desloca ~10 linhas. Por isso, com
`cone_estimator: True`, o corredor é um **estado filtrado**
(`fva_car/corridor.py`, filtro de Kalman estendido iterado):

| Estado | Significado |
|--------|-------------|
| `e` | centro em relação ao eixo dianteiro (esquerda +), m |
| `psi` | rumo do centro no ponto mais próximo do eixo, rad |
| `kappa` | curvatura do centro nesse ponto, 1/m |
| `dkappa` | variação da curvatura por metro (clotoide): entrada e saída de curva |
| `W` | largura do corredor, m |

- **Previsão**: a odometria (`PoseHistory`, eixo traseiro, convertida para o
  dianteiro com `wheelbase`) leva o estado de um quadro ao outro. Assim existe
  centro junto ao eixo, onde a câmera é cega, e entre quadros sem cones.
- **Correção**: cada cone é uma medida "a distância ao centro vale +W/2 (borda
  esquerda) ou −W/2 (direita)". As duas cores entram num só ajuste. A incerteza
  de cada cone vem da homografia: pequena na direção (coluna), grande na distância
  (linha + arfagem). O erro do modelo cresce com a distância.
- **Descarte**: um cone a mais de 3σ da previsão (porta, reflexo, cone de outro
  trecho) fica de fora. Se a rejeição domina dois quadros seguidos, o estimador
  recomeça pelos cones atuais.
- **Estados e velocidade** (fração da vref, nunca abaixo de `curve_min_speed`):

| Estado | Quando | vref |
|--------|--------|------|
| `AMBAS` | cones dos dois lados | ×1 |
| `SO_EXTERNA` / `SO_INTERNA` | uma cor só, na curva (a de fora / a de dentro) | ×0,8 |
| `UMA_BORDA_RETA` | uma cor só, \|κ\| < 0,15 | ×0,85 |
| `MEMORIA` | nenhum cone aceito; segue a previsão por até `memory_m` (0,8 m) / 1 s | ×0,6 e teto de 0,35 m/s |
| `PERDIDO` | além disso | neutro |

- **Liberação**: o critério é a incerteza lateral do centro **1 m à frente**
  (≤ 0,2 m), e não a do eixo. Com só cones distantes, como na partida do ensaio 1,
  `e` e `psi` no eixo são incertos mas correlacionados, enquanto o caminho dos
  próximos metros é conhecido. Com incerteza no eixo acima de 0,1 m, o teto é de
  0,35 m/s até os cones chegarem perto.
- Com o estimador, a tolerância de perda do Stanley cai para 2 quadros / 0,3 s,
  para não somar outro trecho às cegas depois da `MEMORIA`.

Ajustes em `cone_estimator` (dicionário, chaves de `CorridorSettings`), por
exemplo `{"memory_m": 0.6, "row_sigma": 6}`. Para desligar, use
`--no-estimator` ou `"cone_estimator": False`. No vídeo, as bordas do
estimador aparecem em ciano e o centro em branco (tracejados quando sem
validade). O `cones_debug.jsonl` traz `corridor` e `legacy_errors` (o ajuste
antigo, para comparar). O `car.csv` traz `corridor_state` (índice em
`STATES`), `corridor_sigma_e` e `corridor_speed_factor`.

**Replay sem o carro** (só o `cones_debug.jsonl`; dispensa as imagens):

```bash
python3 tools/replay_corredor.py logs/<ensaio> --csv corredor.csv
python3 tools/replay_corredor.py logs/<ensaio> --set row_sigma=6 --set memory_m=0.6
```

Nos ensaios de 23/10: E1 passa de 4 para 186 de 186 quadros válidos, e E2 de 113
para 215 de 246 (os 31 restantes não têm cones). O salto entre quadros caiu de
33 cm / 30° para 0,9 cm / 0,6° (p90 do E2). O custo é de ~1,7 ms por quadro no
notebook; meça no Pi com `tools/benchmark_visao.py --cones`.

### Largura da pista (aprendida em corrida)

A pista muda de largura a cada dia, então `cone_settings.track_width_cm` é só
o **palpite inicial**. Com cones dos dois lados, `W` é medido. A média dos
quadros `AMBAS` vira a **largura aprendida**, que segura `W` quando só uma borda
aparece e sobrevive a uma perda do corredor. Antes de medir, o palpite tem
incerteza de 0,6 m e não puxa `W`. Partindo só de uma borda, o carro segue a
meia largura do palpite, devagar (teto de 0,35 m/s), até ver a outra.

Na pista virtual, com o mesmo palpite de 1,8 m, pistas de 1,0, 1,3, 1,8 e 2,4 m
deram 2–8 cm de erro RMS. A pista que estreita durante a volta também funciona.
A que **alarga** durante a volta (1,1 → 2,2 m) não: a borda esquerda sai da
imagem (câmera 8° à direita), `W` deixa de ser medido e o carro anda até ~35 cm
fora do centro, mas sem tocar cones. A checagem de calibração do detector só
acusa largura fora de 0,4–2,5× o palpite. `cones_summary.json` traz
`learned_width_m`.

## Direção e velocidade com o estimador

- **Pure pursuit** (`steering_law: "pure_pursuit"`, padrão nos cones com o
  estimador). Mira o ponto do centro estimado a `0,4 m + 0,2 s·v` (teto 1,5 m)
  do eixo dianteiro, já deslocado para o caminho do eixo traseiro (em curva, ele
  anda por dentro: R − √(R² − L²)). O caminho junto ao carro vem da clotoide do
  estimador (`path_dkappa` na observação). `--stanley` volta ao Stanley; a
  linha amarela usa sempre Stanley.
- **Prévia de curva** (`speed_preview_m: 2.5`). O carro não freia: em neutro,
  perde 0,55–0,7 m/s² a partir de 1,5 m/s (logs de 20/10). Então a vref cai
  **antes** da curva, para a maior velocidade que ainda permite rolar até cada
  ponto dos próximos 2,5 m com `v² ≤ curve_lateral_accel/|κ|`
  (`coast_deceleration: 0.5`, reação de 0,3 s).
- **Rampa assimétrica** (`speed_rise_m_s2: 1.0`): a vref desce na hora e sobe
  no máximo 1 m/s². Substitui `speed_filter_s` nos cones. `--no-preview`
  desliga a prévia e a rampa.

Na pista virtual (abaixo), com as 4 pistas e 3 sementes, a 0,7 e 1,5 m/s:

| | erro RMS | erro máx. | acel. lateral máx. a 1,5 m/s |
|---|---|---|---|
| Stanley (antes) | 4,4–8,3 cm | 9,6–23 cm | — |
| Pure pursuit | 2,6–4,9 cm | 7–19 cm | 0,30–0,46 m/s² (volta 2–6% mais lenta) |
| Pure pursuit sem prévia | 2,8–4,7 cm | 7,5–18 cm | 0,57–0,65 m/s² (limite 0,4) |

Com erro de calibração (±8 px de linha, 6 px de coluna), visão a 8–10 quadros/s
com 0,2–0,35 s de atraso e servo lento, o pure pursuit continuou com 2,6–6 cm de
RMS, contra 7,5–9,4 cm do Stanley. O maior erro isolado é no grampo de R = 1,8 m
(~18 cm, folga de 58 cm até os cones). O `car.csv` ganha `pursuit_lookahead_m`,
`pursuit_target_y_m`, `path_dkappa`, `preview_speed_m_s` e
`preview_curvature_m_inv`.

### Rápido nas retas, cuidado nas curvas

Nos cones, `velocity_reference` (a velocidade da GUI) é a **velocidade máxima
nas retas**. Quem decide quanto o carro corre em cada trecho:

- **Curva vista** (prévia de 2,5 m): reduz antes dela, para fazer a curva com
  `curve_lateral_accel` (0,4 m/s²).
- **Limite de visibilidade** (`min_curve_radius_m`, padrão 1,2 m, a pista de 10/11; `--min-radius`;
  na GUI, "Curva mais fechada"). A câmera vê ~2,5 m e o carro só rola para reduzir
  (~0,5 m/s²). Então ele supõe que, logo depois do trecho visto, pode vir a curva
  mais fechada da pista, e corre só o que ainda deixa rolar até a velocidade
  dela. Na reta, isso dá ~1,6 m/s.
- **Confiança**: com uma borda só, o teto é de 0,9 m/s (`one_border_speed`); na
  memória ou com incerteza alta, 0,35 m/s; perdido, para.

Na pista virtual, com vref 2,0 contra 0,7 hoje, a volta ficou de 26 a 46% mais
rápida (retas longas: 52 → 30 s). Não houve toques, o erro foi de 2–4 cm RMS e a
aceleração lateral, de 0,31–0,46 m/s². Os piores casos (rolar só 0,35 m/s²,
calibração +8 px, visão a 10 quadros/s com servo lento) também ficaram sem
toques, com erro máximo de 16 cm. Sem o limite de visibilidade, o carro chegava
no grampo com 0,57 m/s².

**No carro, subam aos poucos:** 0,7 → 1,0 → 1,5 → 2,0. Confiram em cada passo se
ele reduz antes das curvas (`preview_speed_m_s` no `car.csv`) e se não abre a
curva. Se ele chegar rápido demais, aumentem a margem: menor raio
(`--min-radius 1.2`) ou `coast_deceleration` menor (o carro rola menos do que se
supôs).

### Curva vista por uma cor só

Em curva, a borda de dentro sai da imagem e o estimador fica em `SO_EXTERNA` ou
`SO_INTERNA` (40–64% do tempo nas pistas com curva da pista virtual). Nesses
estados, a direção pede 15% mais esterço (`one_border_curve_steer_gain`) e o
volante gira 1,5× mais rápido (`one_border_curve_rate_gain`; 120 → 180°/s). Na
pista virtual, 1,1–1,2 foi de neutro a um pouco melhor, inclusive com servo
lento; 1,3/2,0 já passava do ponto. Com 1 o reforço fica desligado.

### Centralização (trim da direção)

Nos ensaios de 10/11, o carro andava 2–12 cm à direita do centro: nas retas, e
sobretudo nas curvas à direita. Nem o Stanley nem o pure pursuit têm termo
integral, então um viés fixo (centro do servo fora, assimetria esquerda/direita)
vira desvio fixo. Agora o erro lateral **em reta e com as duas bordas** é
integrado (`centering_gain` = 0,15 rad por m·s, limite ±4°) e vira **trim do
servo** (`Car.set_steer_trim`). A odometria continua usando o esterço pretendido:
com o centro certo, a roda faz o que a odometria supõe. Se o trim fosse somado
ao comando, o próprio estimador ficaria enviesado (testado na pista virtual).

O trim aprendido é salvo em `logs/direcao_trim.json` no fim de cada corrida (se
houve ao menos 3 s de reta), e a corrida seguinte começa dele. Cada corrida move
o trim salvo só metade do que aprendeu. Na pista virtual, com 3° de viés na
direção, o pure pursuit ficava em média 10 cm de lado (RMS 10,5 cm); da 2ª
corrida em diante, RMS de 3,1–3,7 cm, igual a um carro sem viés. O terminal
mostra o trim no início e no fim. Para desligar: `--no-trim-learning`, ou a
caixa "Aprender o centro da direção" na GUI (aí vale `steering_trim_deg`).

Se o carro continuar fora do centro com o trim já estabilizado, a suspeita passa
a ser a calibração da câmera, porque o carro está centrado no centro *estimado*.
Nesse caso, façam o teste da trena.

### Ultrassom em curva

Em curva, com a roda já esterçada, o ultrassom (que aponta para a frente) vê o
cone da borda de fora muito perto. Antes, isso travava o carro até o fim do
ensaio, embora ele fosse passar ao lado. Agora (`fva_car/obstacle.py`) o eco é
**ignorado** só se tudo isto vale:

1. a visão vê o corredor agora (uma ou duas bordas, ou memória; imagem com
   menos de 0,5 s);
2. há curva no próximo 1,5 m (R ≤ 6,7 m). A roda não precisa já estar virada,
   porque Stanley e pure pursuit vão seguir a curva; só não pode estar virada
   mais de 6° para o lado contrário;
3. o carro está dentro do corredor com folga (|e| ≤ W/2 − 0,18 m);
4. há uma borda do corredor onde está o eco: dentro do feixe (±35°, porque um
   cone perto e de lado ainda ecoa) e a ±25 cm da distância medida, descontado o
   quanto o carro andou desde a foto.

Além disso, um eco curto só para o carro se durar 0,15 s (`obstacle_confirm_s`,
cerca de 2 leituras): nos ensaios de 10/11, 9 de 13 rodadas pararam por uma
leitura isolada falsa de 0,1–0,2 m.

Um obstáculo no meio da faixa não coincide com nenhuma borda e **para como
antes**, assim como em reta, sem corredor válido ou sem esterço. O terminal
mostra `Ultrassom a 0.45 m ignorado: cone da borda direita na curva` (no
máximo 1 vez por segundo), e o `car.csv` traz `obstacle_ignored`. Para parar
por qualquer eco: `--strict-ultrasonic`, ou desmarcar a caixa na GUI.

### Pista virtual (sem o carro)

`tools/pista_virtual.py` roda o **mesmo código do carro** em malha fechada:
`CorridorEstimator`, `RealStanleyFollower`, `PoseHistory` e a configuração do
`main.py`. Simula:
- um carro cinemático: direção com média de 3 amostras e servo; acelera até
  1 m/s² e só rola para desacelerar;
- a odometria do `Car`, com erro de escala e viés de esterço;
- a câmera, pela homografia real: cone visível se a base cai na imagem 640×480
  e a até 3 m; ruído por cone e de arfagem; cones perdidos e falsos; ~15
  quadros/s com 0,12–0,18 s de atraso;
- pistas de retas e arcos (`oval`, `grampo`, `s`, `chicane`), com largura fixa
  ou variando.

```bash
python3 tools/pista_virtual.py                                  # tudo, Stanley x pure pursuit
python3 tools/pista_virtual.py --pista grampo --set velocity_reference=1.5
python3 tools/pista_virtual.py --width 1.0 --width-end 1.6 --row-bias 8   # pista e calibracao piores
python3 tools/pista_virtual.py --pista s --plot s.png                     # trajetorias
```

`--set` muda qualquer chave de `main.parameters`. `--latency`, `--fps`,
`--servo-tau`, `--coast`, `--row-bias` e `--column-bias` pioram o carro, a
câmera ou a calibração. Ela não substitui o ensaio: não modela escorregamento,
reflexos nem a detecção de cor. Serve para comparar ajustes antes de levar ao
carro.

## Iluminação

A pista muda de luz (sombra, janela, lâmpada quente ou fria). O detector
(`tools/cone_path.py`, `auto_light`) leva a cor do piso, a mediana da metade de
baixo da imagem, à cor de referência (`floor_b/g/r`: o piso dos ensaios de
23/10, em que os limiares foram ajustados). Isso é feito com um ganho comum
(brilho) e um ganho por canal (cor da luz), antes da segmentação:
- **Zona morta**: só o excesso além de ±15% de brilho e ±2% de cor é corrigido.
  Na luz de referência a imagem passa intacta.
- **Limites**: ganho de 0,5–3,5× e cor de ±50%. O ganho nunca estoura os
  pixels coloridos mais claros: canal saturado muda o matiz, e o vermelho vira
  laranja.
- Custo de ~0,4 ms por quadro no notebook. Os recortes do classificador saem da
  imagem já normalizada, também na extração do conjunto de treino.

Nos 152 quadros de evento de 23/10, alterados artificialmente:

| Condição | Sem normalizar | Normalizando |
|---|---|---|
| original | 100% | 100% (idêntico) |
| 55% mais escuro | 53% | 93% |
| 70% mais escuro | 5% | 95% |
| 80% mais claro | 89% | 93% |
| luz quente / fria (±25%) | 72% / 58% | 90% / 89% |
| escuro + quente | 78% | 88% |
| contraste (gama 0,6) | 73% | 73% |

(Percentual dos cones detectados no original que continuam detectados.)
`cones_summary.json` traz `light_gain_median` e `light_gain_max`. Ganho perto de
3,5 significa luz fraca demais: aumente a exposição. `tools/ajustar_camera.py`
mostra o piso medido e a correção que seria aplicada. Só troque `floor_*` se
reajustar os limiares de cor sob outra luz.

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
saturação dos cones. Ela salva uma foto por ajuste em `logs/ajuste_camera/<data_hora>/` (vem no Coletar Dados da GUI) e imprime
a linha para colar em `main.py`, por exemplo
`"camera_settings": {"fourcc": "MJPG", "exposure": 150, "auto_wb": False}`.
Com `camera_settings` vazio, a câmera é **posta** no modo automático (exposição
e balanço de branco) ao abrir. A webcam guarda o último ajuste até ser
desconectada; antes, a missão herdava a exposição manual deixada pelo
`ajustar_camera.py`. A ferramenta também volta a câmera ao automático no fim,
mede a linha "auto" com o automático ligado de verdade, testa exposições até
120 ms (salas escuras) e aceita a partir de 15 fps (a visão processa 8–15
quadros/s no Pi). Se o piso não passa de V = 25/255 nem no automático, ela
avisa que está escuro demais para detectar cones.

Para separar o custo do detector da disputa com as demais threads, rode no Pi
`python3 veiculo_real/tools/benchmark_visao.py --cones` (sem `--image` usa a `vision_camera.jpg` mais recente em `logs/`)
e compare com o `detection_s` do log.

Os logs acrescentam `cone_red_count`, `cone_green_count`, `corridor_width_m`,
`cone_green_left`, `cone_estimated`, `cone_branch`, `cone_segment` (0 reta,
+1 esquerda, −1 direita), `line_hold`, `line_lost_frames` e `hold_speed_factor`.
`line_valid`, `cross_track_m`, `heading_error_rad` e `curvature_m_inv` continuam
representando a referência usada pelo controlador, agora o centro do corredor.

**ROI (horizonte fora).** A primeira linha útil sai da homografia: o ponto a
`max_range_cm` (3 m) mais 1,5 altura de cone. Na calibração atual é a linha 247
de 480, então ~51% da imagem (parede, portas, placas) nunca entra na
segmentação, na correção de luz nem na escolha de cones. A **transmissão**
(`--stream`) também manda só essa região, com a legenda numa faixa preta
acima, em vez de escrita sobre a imagem: 640×323 em vez de 640×480 e ~17 kB em
vez de ~24 kB por quadro (JPEG 70). Os quadros brutos (`--record-frames`) e os
de evento continuam inteiros, porque a rotulagem e a calibração usam a imagem
completa.

A transmissão mostra a contagem de cones aceitos por cor, inclusive quando
a geometria do corredor é inválida. Regiões unidas por sobreposição podem
fornecer a base inferior visível; bases ocultas não são inventadas.
