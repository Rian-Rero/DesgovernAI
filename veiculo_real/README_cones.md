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
- **Faixa de busca** (`band_m` 0,2 + `band_growth` 0,1 por m além de 1 m: 0,3 m
  a 2 m, 0,4 m a 3 m). Longe, o portão de 3σ é largo (passa de 1,5 m a 2,5 m),
  porque a curva à frente ainda é incerta. Por isso, depois do ajuste com os
  cones aceitos, um cone a mais que a faixa da borda **ajustada** sai, e o
  quadro é refeito sem ele. Uma curva de verdade desloca todos os cones longe
  juntos, e o ajuste os acompanha; um objeto isolado (mochila, roupa) fica de
  fora. Na pista virtual com 15 objetos fixos 20–80 cm fora da pista, de cor ao
  acaso:

  | | voltas completas | erro RMS (médio / pior) | toques |
  |---|---|---|---|
  | sem faixa | 11 de 12 | 11,1 / 41 cm | 2 |
  | com faixa | 12 de 12 | 4,4 / 6,6 cm | 0 |

  Com faixa, o resultado ficou igual ao de uma pista sem objetos. Nos 11 ensaios
  reais de 10/11, ela tira 22 de ~11 mil cones, sem perder quadros válidos. Não
  resolve objetos da cor **errada** colados na borda oposta, muitos e logo no
  início (a largura "encolhe" ao recomeçar): isso é trabalho do classificador.
  `{"band_m": 0}` desliga. Os cones tirados entram em `rejected`.
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

### Profundidade dos cones longe

A distância de cada cone sai da **linha da base** na imagem (homografia do
chão). Com a câmera quase nivelada, 3 m fica a ~5 linhas do horizonte: 1 px
de linha vale 0,8 cm a 1 m, 3 cm a 2 m e 7 cm a 3 m na imagem cheia (640 px), e
o dobro na imagem reduzida que o detector usa (`vision_width` 320). O que foi
medido (novembro de 2026):

- **Arfagem não é a causa.** Estimando a inclinação da câmera quadro a quadro
  junto com o corredor, os ensaios de 10/11 deram ~1 px de desvio, e na pista
  virtual não mudou nada. A ideia foi descartada.
- **A resolução reduzida é.** Em cones sintéticos na posição exata (calibração
  real, borda suave, desfoque), o erro de distância a 2-3 m foi o dobro a 320 px.
  Nos 411 quadros reais de 10/11, a 320 px a distância dos cones a 1,7-2,6 m
  diferia da de 640 px em 4-7 cm (mediana) e 8-14 cm (p90).
- **Refino da base na imagem cheia** (`refine_bases` em `fva_car/cones.py`,
  `cone_refine_bases` no `main.py`). Para cada cone a mais de 1,3 m, a base é
  refeita na imagem de 640 px, numa janela pequena em volta dele, com a mesma
  segmentação de cor e a mesma regra de base do detector. Nos quadros reais, a
  diferença para 640 px caiu para 0 (mediana) e 3,6-4,9 cm (p90) a 1,7-2,6 m.
  Custo: +0,7 ms por quadro no notebook (5,6 → 6,3 ms; 640 inteiro custa 14,5).

O estimador já pesa cada cone pela incerteza da homografia (pequena na direção,
grande na distância, crescendo com o quadrado da distância), então cones longe
pesam menos. Com a base mais precisa, eles atrapalham menos o ajuste da curva.

### Hierarquia: cones perto pesam mais

O estimador ajusta as bordas por mínimos quadrados ponderados: cada cone pesa
pelo inverso da sua variância, que cresce com a distância. Há dois termos: a
incerteza da homografia (quadrática; 1 px de linha vale ~0,8 cm a 1 m e ~7 cm a
3 m) e um erro de modelo linear (`model_growth`, 0,10 m por m desde 19/11;
antes 0,05). Peso de um cone de borda numa reta, relativo ao de 1 m:

| distância | 0,8 m | 1 m | 1,5 m | 2 m | 2,5 m | 3 m |
|---|---|---|---|---|---|---|
| desvio do cone | 13 cm | 15 | 20 | 25 | 31 | 36 |
| peso (0,10/m, atual) | 1,34 | 1 | 0,56 | 0,36 | 0,25 | 0,18 |
| peso (0,05/m, antes) | 1,25 | 1 | 0,63 | 0,43 | 0,31 | 0,24 |

O Stanley usa o deslocamento e o rumo junto ao eixo, que saem principalmente
dos cones perto. Os longe entram com pouco peso, mas não saem: com cones a ~1 m
uns dos outros, são eles que mostram a curva chegando.

Na pista virtual (cones a 1 m, 7 quadros/s), com ruído e erro de calibração
nos cones longe (o caso real), 0,10/m tirou os toques (2 → 0) e o pior erro
(59 → 35 cm). 0,2/m ou mais piorou (voltas incompletas): a curva à frente só
aparece nos longe. Nos ensaios reais reprocessados, o custo é um pouco mais de
velocidade lenta (21% → 28% dos quadros válidos; partida mediana 0,73 → 0,90 s).

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

### Stanley (padrão desde 19/11)

O professor pediu o Stanley. Ele roda no centro estimado, em metros no chão (os
cones vêm projetados no piso pela homografia: é a "vista de cima" numérica):

    esterço = rumo + atan(k·e / (ks + v)) + atan(L·κ)

- `e` e `rumo`: deslocamento e rumo do centro estimado junto ao eixo dianteiro;
  `κ`: curvatura (termo de regime, `stanley_curvature_ff`).
- **k e ks fixos** (`stanley_gain` 2,5, `stanley_softening` 0,3).
- **v = velocidade MEDIDA, filtrada** (`stanley_use_vref: False`,
  `stanley_speed_filter_s` 0,3 s, passa-baixa de 1ª ordem). Antes era a vref:
  na partida, nas curvas e freando, o carro está mais lento que a vref, e o
  termo ficava fraco justamente ali. A do encoder sem filtro oscila e faria o
  ganho efetivo oscilar junto.
- **Escolha de k**, na pista virtual com a pista de 17/11 (largura 1,3 m, cones
  a 1 m, visão a 7 e 11 quadros/s): k 1,5 → 7,8 cm RMS; k 2,5 → 5,8 cm; sem
  toques. No simulador k 3 ainda melhora, mas o carro real é mais ruidoso
  (poucos cones por quadro). `--stanley-gain` ou o campo "Ganho Stanley k" da
  GUI ajustam na pista: oscila, baixe; passa reto da borda, suba.
- **Taxas** (pergunta do professor): o Stanley roda no laço de controle a
  **~48 Hz**, mas a imagem nova chegava a **6–8 por segundo**, com **~280 ms**
  de atraso. Cada imagem serve ~7 passos; entre elas o erro é levado ao instante
  atual pela odometria (compensação do atraso). O gargalo é a visão, não o
  controle; o 📋 Relatório mostra essas taxas em cada volta.

### Pure pursuit (opcional)

- **Pure pursuit** (`--pure-pursuit`, ou a caixa na GUI). Mira o ponto do centro estimado a `0,4 m + 0,2 s·v` (teto 1,5 m)
  do eixo dianteiro, já deslocado para o caminho do eixo traseiro (em curva, ele
  anda por dentro: R − √(R² − L²)). O caminho junto ao carro vem da clotoide do
  estimador (`path_dkappa` na observação). A linha amarela usa sempre Stanley.
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

### Partida (sem "engasgo")

Nos 11 ensaios de 10/11 o carro ficava 0,9-3,9 s preso a 0,35 m/s depois de
ligar, e a velocidade oscilava entre 0,03 e 0,4 m/s. Havia três causas:

1. **Velocidade de recuperação pelo eixo.** O estimador mandava ir devagar com
   sigma do deslocamento **no eixo** > 0,1 m. Na partida os cones estão a 1,5 m
   ou mais, e esse sigma só cai andando. Agora vale a incerteza do caminho **a
   frente** (`slow_sigma_lateral`, sigma lateral a 1 m < 0,1 m) ou um eixo muito
   incerto (`slow_sigma_e` 0,3 m), ou uma borda só com a largura ainda nunca
   medida. Na **primeira** inicialização da corrida, o palpite do deslocamento e
   do rumo no eixo é 0,5 m / 0,5 rad (`start_sigma_e/psi`; depois de uma perda,
   0,6), porque o carro foi posto à mão no corredor. No replay dos mesmos
   ensaios, a espera foi de 1,6-4,1 s (média 2,7) para 0-1,9 s (média 0,5). Em 4
   deles, o critério antigo também segurava o carro em 70-100% do resto da volta.
2. **Zona morta do ESC.** Em regime, o acelerador fica em ~0,4 a 0,3-0,6 m/s:
   o carro só anda acima de ~0,35. O feedforward era `0,17*vref` (0,06 a 0,35
   m/s), e o integrador levava 1-2 s para chegar lá, oscilando em volta da
   zona morta (o motor ligava e desligava). Agora o ponto de partida é
   `0,32 + 0,22*vref` (`THROTTLE_FF_OFFSET` e `THROTTLE_FF_GAIN` em
   `fva_car/car.py`), e o carro **aprende** o acelerador de regime por faixa de
   0,1 m/s (`fva_car/throttle.py`). Em regime (na velocidade pedida, sem saturar
   e sem acelerar), o acelerador aplicado é o que aquela velocidade pede; o
   feedforward passa a sair dessa tabela. Ela é salva entre corridas em
   `logs/acelerador.json` (como o trim; a limpeza da GUI não apaga), com pelo
   menos 3 s de regime. Assim outro carro, ESC ou carga de bateria se ajusta
   sozinho. Um valor fixo (`--throttle-offset`, ou um número no campo da GUI)
   desliga o aprendizado, para comparar.
3. **Alvo do pure pursuit no ponto cego.** Com o eixo ainda incerto
   (`near_sigma_e` 0,1 m), o alvo fica a pelo menos `check_distance_m` (1 m),
   onde o caminho é conhecido, e não a 0,4 m. Antes a direção batia no fim de
   curso parada em 6 de 11 ensaios. Além disso, a rampa da vref parte da
   velocidade medida, e o portão de salto lateral se alarga por 3 sigma
   enquanto o estimador converge.

Na pista virtual, cada volta ficou ~1 s mais rápida nas 6 pistas, com o mesmo
erro lateral e sem toques.

**Ajustar e comparar na pista** (aba da missão da GUI, ou linha de comando):

| GUI | `main.py` | o que faz |
|---|---|---|
| Acelerador mínimo: `auto` (padrão) | (nada) | o carro aprende e salva em `logs/acelerador.json` |
| Acelerador mínimo: um número | `--throttle-offset 0.28` | fixo, sem aprender: tranco na partida, baixe; o motor liga e desliga, suba |
| Partida nova (desmarcada) | `--old-start` | volta à partida antiga, inclusive o acelerador |
| Refino da base dos cones longe (desmarcado) | `--no-refine` | base dos cones longe na imagem reduzida |

O nome do log ganha `_acel0.28`, `_partida-antiga` ou `_sem-refino`, e o
📋 Relatório mostra as opções e a seção de partida de cada volta.

### Atraso da visão

Nos ensaios de 10/11 a imagem chegava ao controle com 260–290 ms (p90 até
420 ms), a 6–9 quadros/s. A 1 m/s, são ~30 cm andados entre a foto e o uso.

- **Visão a 320 px** (`vision_width: 320`, padrão também na GUI). Em 617
  quadros reais, a 320 px os cones achados até 1 m / 1–2 m / 2–3 m foram 98 /
  94 / 86% dos achados a 640 px, com a detecção 2,5× mais rápida. A 480 px:
  99 / 97 / 91%, só 28% mais rápida. `--vision-width 640` volta ao antigo.
- **Compensação exata** (`exact_latency_compensation`, ligada): o estado do
  estimador (clotoide inteira) é levado do instante da foto até o do comando,
  pela velocidade e guinada atuais, até 0,5 s. Antes: curvatura constante, até
  0,25 s.
- **Velocidade pelo atraso** (`latency_speed_budget_m: 0.5`, era 0,3 até
  24/11): v ≤ 0,5 m ÷ idade média da imagem. Com os 220 ms dos ensaios de 19/11,
  isso dá ~2,3 m/s (com 0,3 eram 1,4); com a visão mais rápida, o teto sobe
  sozinho. Na pista virtual (10 q/s, 220 ms), 0,5 teve pior erro de 22 cm
  contra 37 com 0,3, e nenhuma volta a menos. 0 desliga.
- **Gravar quadros** (`--record-frames`) deixa a visão ~20% mais lenta: use só
  em coleta.

**De onde vem o atraso** (25 voltas de 19/11 à noite, medianas, carro andando):

| trecho | ms |
|---|---|
| espera do `read()` pelo próximo quadro (contava como idade) | 38 |
| processamento (detecção 66 + geometria 16 + resto) | 89 |
| fila + pipe até o controle + laço de 48 Hz pegar | 42 |
| **idade quando o quadro chega ao controle** | **176** |
| idade média durante o uso (o mesmo quadro serve ~5 passos) | 225 |

O detector leva ~6–7 ms por quadro no notebook e ~89 ms no Pi (12×). O custo
está nos laços em Python, não no OpenCV. Mudanças de 24/11:
- **Horário real da foto** (`fva_car/vision_process.py` `frame_time`). Antes
  era o horário de antes do `read()`, que inclui a espera por um quadro que nem
  existia ainda. Agora vale o horário que o driver da câmera marca quando o
  quadro chega (V4L2, mesmo relógio de `time.monotonic`). Fora do plausível
  (ausente, no futuro ou mais de 0,5 s antes), volta ao antigo. A compensação
  do atraso e a idade usam esse horário. O `vision_status.json` mostra
  `capture_clock` (`v4l2` ou `leitura`), e `vision_capture_s` no `car.csv`
  passa a ser o atraso do driver até o fim da leitura.
- **Imagem por memória compartilhada** (`SharedFrames`). Cada observação
  levava o quadro de 900 KB pelo pipe: 2,9 ms no notebook contra 0,4 ms sem ele
  (no Pi, mais). O `send` travava o processo de visão até o controle ler, e a
  cópia disputava o GIL do laço de controle. Agora o quadro vai para um anel de
  4 vagas compartilhadas e pelo pipe só vai o número da vaga.
- **SLAM vetorizado**. Antes, a associação calculava um marco por vez em
  Python: 1,4 ms por quadro no notebook. Vetorizada, dá o mesmo resultado
  (diferença de 1e-13) e o mapa custa 0,3 ms por quadro.

**Saúde do Raspberry e núcleos** (24/11, `fva_car/system_health.py`). O Pi 4
reduz o clock sozinho quando esquenta (a partir de ~80 °C) ou com subtensão
(bateria fraca), e isso não aparecia em lugar nenhum. A diferença de 12× para o
notebook é maior que a usual entre os dois.
- **Saúde:** a cada 2 s, a volta lê a temperatura, o clock e o
  `vcgencmd get_throttled`. O terminal mostra o estado na partida e o pior da
  volta no fim, com avisos (aquecimento, subtensão, clock caindo sem o governor
  `performance`). Isso fica em `saude_raspberry.json`, no `cones_summary.json`
  e na linha `raspberry:` do relatório do ensaio. `health_monitor: false`
  desliga.
- **Núcleos** (`cpu_cores`): o processo do controle (com gravação e
  transmissão) fica nos núcleos 0–1, e o da visão (captura + detecção) nos
  núcleos 2–3. Assim um não rouba o núcleo do outro. Vale só com a visão em
  processo e 4+ núcleos. `--sem-nucleos` ou a caixa "Controle e visão em
  núcleos separados" na GUI devolvem a escolha ao sistema, para comparar.

Se o relatório acusar limitação: dissipador/ventoinha no Pi e bateria
carregada. Para fixar o clock no máximo:
`echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
(volta ao normal ao religar).

**Próximo passo, se ainda faltar velocidade:** o resto do atraso é
processamento em Python no Pi. Dá para rodar dois processos de detecção
alternando quadros (dobra a taxa, não reduz o atraso de cada um) ou levar os
laços de `tools/cone_path.py` (`_bases`/`contact`) para operações do OpenCV.

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

**Até 4 m/s** (24/11): `--speed` aceita até 4,0 e a GUI tem 3,0 e 4,0. O `Car`
corta o acelerador só acima de 4,5 m/s; o rádio continua indo até 2 m/s com o
manche cheio (`RC_VELMAX`).

**A trava de verdade estava no servo do acelerador.** `fva_car/servos.py`
cortava `velmax` em 1,5 m/s, e o `Car` nem passava a velocidade. Com isso, o PWM
máximo era 113° (18° acima do neutro), e com o acelerador em 100% o carro não
passava de ~1,5 m/s (logs de 19/11: máximo 1,56 m/s; saturado, 1,0–1,2 m/s).
- **Agora:** o `Car` cria os servos com `velmax` 4,5 m/s (PWM até ~146°).
- **A unidade do comando não mudou:** `u = 1` continua sendo os mesmos 18°. Por
  isso o feedforward (0,32 + 0,22 v) e o mapa aprendido (`logs/acelerador.json`)
  valem como antes.
- **Acima de ~1,5 m/s o comando passa de 1**, até 2,8 no automático. No rádio,
  o comando continua parando em 1. Na prática, quem decide a velocidade é o menor de:

| limite | valor hoje | de onde vem |
|---|---|---|
| teto pedido (GUI / `--speed`) | até 4,0 m/s | nas retas |
| curva (`curve_lateral_accel` 0,4 m/s²) | √(0,4·R): R = 2 m → 0,9 m/s | prévia de 2,5 m |
| atraso da visão (0,5 m ÷ idade) | 220 ms → ~2,3 m/s | `latency_speed_budget_m` |
| parada no alcance do ultrassom | **~2,8 m/s** | `ultrasonic_speed_cap` |

O último é segurança: a 4 m/s, a distância de parada é 6,6 m (0,3 s de reação +
0,15 s de confirmação + frenagem de 1,75 m/s²), mais que os 4 m que o ultrassom
alcança. Acima de ~2,8 m/s, um obstáculo só seria visto quando já não desse para
parar. `--sem-teto-ultrassom` (caixa "Teto pelo ultrassom" na GUI) tira o teto,
só em pista fechada e sem ninguém na frente. Na pista virtual (10 q/s, 220 ms),
com o teto em 2,8 m/s, a velocidade média foi de 0,56 para 0,88 m/s na pista
de 1,3 m, sem toques novos. As curvas seguram o resto. Para ir além, a visão
precisa ficar mais rápida (menos atraso).

**No carro, subam aos poucos:** 0,7 → 1,0 → 1,5 → 2,0 → 3,0 → 4,0. Confiram em cada passo se
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
vira desvio fixo. Agora, **em reta e com as duas bordas**, o carro aprende o
**trim do servo** (`Car.set_steer_trim`, limite ±4°) pelo **esterço médio**
comandado (`centering_steer_gain` = 0,3/s). Seguindo uma reta, a roda real fica
em 0, então o comando médio é −(viés + trim), e o trim anda até ele zerar.

Por que não pelo erro lateral (o modo antigo, `centering_gain` = 0,15, agora
0)? Sem a IMU, a guinada da odometria vem do esterço comandado. Com viés na
direção, a odometria entorta, e com ela o estimador. Na pista virtual, com 2° de
viés, o carro ficava 11 cm fora do centro enquanto o estimador o via a −2 cm,
com o sinal trocado. O erro estimado não vê o viés; o esterço médio vê. Em 4
pistas com ±2° de viés e visão a 7 q/s, o erro RMS ficou em 3,8–8,7 cm pelo
esterço, contra 5,2–12,3 cm pelo erro lateral. Com ganho 1,0 o carro oscilou na
pista fechada. A odometria continua usando o esterço pretendido:
com o centro certo, a roda faz o que a odometria supõe. Se o trim fosse somado
ao comando, o próprio estimador ficaria enviesado (testado na pista virtual).

O trim aprendido é salvo em `logs/direcao_trim.json` no fim de cada corrida (se
houve ao menos 3 s de reta), e a corrida seguinte começa dele. Cada corrida move
o trim salvo só metade do que aprendeu. Na pista virtual, com o trim já certo,
2° de viés dão o mesmo erro de um carro sem viés (4,0 cm RMS). O terminal
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

Leitura de 0,00 m (abaixo de 3 cm) é **falha do sensor**, não obstáculo: conta
como leitura inválida (`ULTRASONIC_MIN_VALID_M` em `main.py`).

Um obstáculo no meio da faixa não coincide com nenhuma borda e **para como
antes**, assim como em reta, sem corredor válido ou sem esterço. O terminal
mostra `Ultrassom a 0.45 m ignorado: cone da borda direita na curva` (no
máximo 1 vez por segundo), e o `car.csv` traz `obstacle_ignored`. Para parar
por qualquer eco: `--strict-ultrasonic`, ou desmarcar a caixa na GUI.

### Linha amarela com a câmera alta

Com a câmera apontada para baixo, a linha aparecia "em pé" na imagem. Com a
câmera alta (a dos cones), ela deita e fica comprimida perto do horizonte. Nas
curvas fechadas o detector antigo, que procura uma coluna por linha da imagem,
não a acha. Além disso, o ajuste por parábola ou arco errava o rumo, às vezes com
o sinal trocado. Agora (`line_birdseye`, `line_estimator`, ligados por padrão):

- **Vista de cima** (`BirdsEyeView`): a homografia leva a imagem a uma vista do
  chão de 0,5 a 3 m à frente e ±1,0 m para os lados, a 1 cm por pixel (200×250).
  A borda do campo de visão (9 px) fica de fora: o canto preto da projeção
  criava falsas linhas.
  Nela a linha volta a ficar como na câmera inclinada, e o mesmo detector e os
  mesmos `path_settings` funcionam (sem ROI; extensão mínima de 15%). A
  conversão leva ~6 ms por quadro. A transmissão mostra essa vista de cima.
- **Estimador**: os pontos da linha (um a cada 15 cm) entram no estimador do
  corredor como linha central (estado `LINHA`), com clotoide e odometria. Daí
  saem pure pursuit, prévia de curva e limite de visibilidade, como nos cones.
  Na curva fechada à esquerda a linha sai da imagem (câmera 8° à direita); a
  memória vai até 2,5 m / 5 s, com a curvatura mais estável que nos cones.

Nos quadros reais de 10/11 (fita fina e tracejada, piso com reflexo quente do
mesmo matiz), três filtros foram necessários:

- **Cones apagados** antes da vista de cima (`mask_cones`): verde e vermelho
  viram a cor mediana do piso. Os riscos dos cones esticados pela homografia
  eram tomados pela fita.
- **Saturação mínima** da fita achada (`LINE_MIN_SATURATION` = 36; fita ~49,
  reflexo ~30): abaixo disso o quadro conta como "sem fita" e a memória segue.
- **Matiz até 35** (`h_max`): o laranja dos cones e do piso fica de fora.

Com isso, a fita é achada em 53% dos quadros e 86% delas estão certas (antes:
97% achadas e só 46% certas: seguia o reflexo).

**Fita junto com os cones** (`cone_line_fusion`, ligado): no modo cones, se
houver fita no centro, um quadro sim e um não ela entra no estimador do
corredor como linha central, junto com os cones como bordas. Sem fita
confiável, nada muda. Custa ~6 ms por quadro em média no notebook. O debug
traz `line_points` no corredor.

Na foto do corredor com uma fita sintética: as retas (com até 25 cm de
deslocamento e rumo de 0,2 rad) erram milímetros. Rodando a pista com curvas de
1,25 m, as imagens tiveram 75/75 quadros válidos e erro lateral mediano de 1 cm
(máximo de 4,6 cm). Na pista virtual (`--linha`), o pure pursuit teve 1,4–3,8 cm
de RMS, contra 5,4–7,5 cm do Stanley. Para voltar ao modo antigo (câmera
inclinada para baixo): `--line-classic`, ou desmarcar "Linha: câmera alta" na GUI.

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
câmera ou a calibração. `--objetos N` põe N objetos fixos 20–80 cm fora da pista,
de cor ao acaso, que o detector toma por cone (mochila, roupa). `--falsos`
define a fração de quadros com um falso ao acaso (padrão 0,05). Com `--linha`, a
pista usa os ajustes do estimador da linha, como o `main.py`. Ela não substitui o ensaio: não modela escorregamento,
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

A calibração atual tem pontos de 0,75 a 1,65 m e ±0,3 m. Os cones ficam a
±0,5–0,65 m e são vistos até 3 m, ou seja, fora da área medida. Pela fita
amarela dos ensaios de 10/11, o centro achado pelos cones fica a 1–4 cm da
fita a 1 m (`tools/analisar_ensaio.py --linha`). Ainda assim, vale refazer a
calibração com a grade:

**Cubra a faixa onde os cones são usados.** A homografia só vale onde houve
pontos medidos; fora disso ela é extrapolada. A calibração de 05/11 tinha 12
pontos de 0,75 a 1,65 m e ±0,3 m. Nos ensaios de 10/11 os cones até ~1,4 m
erravam ~2 cm, e além de 1,5 m, ~20 cm, mas o carro usa cones até 3 m e a
±0,6-0,9 m. O `main.py` agora avisa (`AVISO calibracao`) quando os pontos não
chegam a 80% do alcance (2,4 m) ou a ±0,6 m.

1. Na GUI, aba de testes: **🖨 Marcadores da grade**. Gera e abre
   `calibracao/marcadores_grade.pdf` (`tools/grade_calibracao.py`):
   - página 1: mapa visto de cima com os pontos que a câmera **vê** (previstos
     pela calibração atual), numerados, e como medir;
   - depois: um marcador xadrez por ponto, com número e coordenadas. O ponto é
     onde os quadrados pretos se tocam.

   No piso brilhante, papel branco e preto fosco aparece onde fita e caneta somem
   no reflexo. Imprima em tamanho real. Os marcadores de 2 m em diante são
   maiores. A câmera de 11/2026 está virada um pouco para a esquerda: dos 30
   pontos da grade fixa, só 20 apareciam, e quase nenhum perto. A grade
   sugerida tem 27, todos visíveis (a cada 25 cm onde a imagem é estreita).
2. Com o carro parado e as rodas retas, estique uma trena do **centro do eixo
   dianteiro** reta para a frente e marque os x. Em cada x, meça os y na
   perpendicular (esquadro ou triângulo 30/40/50 cm). ESQ = +. Cole cada
   marcador com o centro sobre o ponto.
3. **📷 Foto p/ calibração**. O carro tira a foto (sem janela no Pi), a GUI a
   traz para `calibracao/fotos/` e abre a calibração manual no notebook.
4. Clique nos pontos medidos e digite `x y` em metros. Para usar a grade
   sugerida com a mesma numeração do PDF, abra a foto com `--grade`.
   Nesse modo, a janela pede um ponto por vez, na ordem:
   - o círculo magenta mostra onde o marcador deve estar (pela calibração
     anterior), e os cinza, os próximos;
   - a **lupa** à direita amplia 5× em volta do mouse;
   - clique no centro, ou use **Pular** (espaço) se ele não aparece; Backspace
     desfaz;
   - com 5 ou mais pontos, uma cruz azul mostra a previsão pelo ajuste novo;
   - cada ponto ganha uma cor pelo erro **deixando-o fora do ajuste**: verde
     até 3 cm, amarelo até 6, vermelho acima (clique ou medida errada).

   `--grade-fixa` volta à grade antiga de 30 pontos.
5. Salvar grava `calibracao/camera_lateral.json`, com o erro de cada ponto.
   Depois, **Enviar arquivos** leva o arquivo ao carro.

Abra uma foto existente com `python3 tools/calibrar_camera_lateral.py --image foto.jpg`
para digitar as medidas. A homografia vale para o
contato no piso, não para o topo dos cones.

## Coleta, limpeza e relógio do carro (aba de dados da GUI)

- **📥 Coletar dados** é incremental. Um `find` no carro lista tudo numa ida só,
  e só vem o que falta no PC, num único fluxo `tar` (milhares de frames sem uma
  ida e volta por arquivo). O que já foi baixado fica anotado em
  `<destino>/<carro>/.coletados.json`. Mover ou reorganizar ensaios no PC não
  faz a coleta baixar de novo. O log mostra quantos arquivos há no carro,
  quantos já estão no PC e quantos MB vão vir. "Baixar de novo o que já foi
  coletado" força tudo.
- **🗑 Apagar do carro o que já está no PC…** coleta primeiro, depois mostra
  quantos arquivos e MB saem e pede confirmação. Só apaga o que está no PC com o
  mesmo tamanho. Nunca apaga `direcao_trim.json` (o trim aprendido) nem
  arquivos mexidos no último minuto (corrida em andamento). Pastas vazias saem
  junto, e o log mostra o espaço livre no carro.
- **🕒 Acertar relógio do carro**: o Pi não tem bateria de relógio e, sem
  internet no hotspot, liga com a hora errada, e os ensaios ganham nomes de
  data errados. O botão acerta data, hora e fuso pelo notebook (`sudo`, com a
  senha da GUI) e salva no `fake-hwclock`, para o próximo boot começar perto
  da hora certa. Toda execução do `main.py` pela GUI confere o relógio e acerta
  sozinha se a diferença passar de 2 s.

## Ensaios de 17/11: o que deu errado e o que mudou (19/11)

19 voltas com Stanley, ~29 s cada. O carro batia em cones e não seguia bem a
trajetória. Nos logs:

| achado | evidência | correção |
|---|---|---|
| Faixa amarela do corredor virou o "centro" | estado LINHA em 1–16% dos quadros; ex.: largura 0,5 m, rumo +19°, direção no batente (`tools/vista_de_cima.py` mostra) | fusão com fita **desligada** (`cone_line_fusion: False`; `--fita-amarela` liga) |
| Visão lenta | 6–8 quadros/s, 280 ms; a geometria custava 124 ms nos quadros com a fita contra 19 ms sem | idem: no Pi, ~142 → ~85 ms por quadro (estimado pelos tempos do próprio log) |
| Poucos cones | 0–2 aceitos por cor e quadro (cones a ~1 m; câmera baixa e quase nivelada) | tamanho real do cone e tolerâncias pela matriz de confusão (abaixo) |
| Parado com cones à vista | ficou até 9 s parado vendo 2 cones (recomeço exigia 3) | com a largura já aprendida, **2 cones** recomeçam |
| Parou no meio da curva | memória de 1 s venceu com 22 cm andados (curva lenta) | `memory_s` 2,5 s (o limite de distância continua) |
| Calibração de 17/11 | fileira de 1,5 m inteira 19 cm deslocada (desvio 0,7 cm entre os pontos) | fileira descartada (guardada em `pontos_descartados`); homografia ajustada **em pixels** (erro fora do ajuste 6,5 → 3,2 px) |
| Origem em x na frente do pneu | a calibração foi medida com o carro encostado na parede (frente do pneu), mas o carro usa o eixo dianteiro, onde ficam a câmera e o ultrassom, 7 cm atrás | 0,07 m somados a todos os x (`--da-frente-ao-eixo 0.07`); ultrassom no eixo (`SENSOR_X_M` 0) e os 7 cm do pneu na distância de parada (`FRONT_OVERHANG_M`) |
| Carro ~15 cm à esquerda do centro | a calibração punha a câmera (que está no centro do carro) 18 cm à esquerda do zero da trena; reprocessando os ensaios, o centro real da pista ficava 13–19 cm à direita do carro (mediana −12,7 cm), com o carro achando que estava centrado | origem lateral movida para a câmera (`--revisar ... --origem-na-camera`); folga de cada lado volta de ~34 para ~52 cm |

**Cone real** (18,5 cm de altura, 10,5 cm de base; antes supunha 20 × 15). As
tolerâncias de tamanho foram escolhidas pela matriz de confusão contra os 3.779
rótulos humanos (`relatorios/matriz_confusao_detector.png`):

| detector | precisão | revocação |
|---|---|---|
| antes (20 × 15 cm, pé até 2,2×) | 0,952 | 0,731 |
| só o tamanho real (mesmas tolerâncias) | 0,953 | 0,683 |
| agora (18,5 × 10,5, pé até 4×, altura até 3,5×) | 0,948 | 0,750 |

Na imagem reduzida, o borrão, a sombra e o reflexo deixam a mancha do cone até
3–4× mais larga que a base: com o tamanho real e as tolerâncias de antes,
cones reais eram descartados. A rede usa sempre o recorte do treino
(`cone_height_m` no `.json`), independente do tamanho do detector.

**Rede** (`relatorios/matriz_confusao_rede.png`): validação cruzada por ensaio
(5 rodadas, cada ensaio fora do treino em uma), a medida honesta.

| modelo | cones verdes mantidos | vermelhos mantidos | não-cones descartados |
|---|---|---|---|
| só 10/11 (3.779 recortes, 11 ensaios) | 96% | 97% | 74% |
| a de 10/11 testada nos 2.621 recortes de 17/11 (lugar novo) | 99,8% | 91,5% | 82% |
| **10/11 + 17/11** (6.400 recortes, 29 ensaios; a do carro) | **98%** | **97%** | **77%** |

A rede antiga perdia 8,5% dos cones vermelhos no corredor de 17/11; com os
rótulos de lá, perde 3%. `treinar_classificador_cones.py --validacao-por-ensaio 5
--matriz m.png` refaz.

**Diagnóstico da calibração** (`calibrar_camera_lateral.py`, ao salvar e com
`--revisar arquivo.json`): a linha do chão logo abaixo da câmera aparece
vertical na imagem; o y dela é a posição lateral da câmera no referencial das
medidas. Com a câmera no centro do carro, tem de dar ~0 (10/11: −0,02 m; 17/11:
+0,18 m, a trena não saiu do centro do eixo). Também mostra o giro (a câmera
está ~10° para a direita: não é erro, a homografia o considera, só tira visão
do lado esquerdo perto do carro) e avisa de fileiras inteiras deslocadas.
`--origem-na-camera` e `--tirar-fileira X` corrigem e gravam.

**Câmera (recomendação, sem código):** em 17/11 ela estava baixa e quase
nivelada, com metade da imagem em parede e teto. O chão útil (0,75–3 m) ocupava
só ~130 linhas (em 10/11 eram ~170), e cada linha a 3 m vale ~7 cm. Inclinar a
câmera para baixo (horizonte perto do topo da imagem) dá mais linhas por metro:
distância mais precisa e mais cones perto. Depois, recalibrar.

**Para os próximos testes:** os 2.621 recortes de 17/11 já estão na fila de
rotulagem (🏷 na GUI); rotular e retreinar adapta a rede ao lugar novo.

### Ferramentas novas

- `tools/matriz_confusao.py <rotulagem> [--variante NOME AJUSTE=VALOR ...]
  [--modelo m.onnx] --png m.png`: detector (e rede) contra os rótulos humanos.
  Cada item guarda a calibração do seu dia (a câmera muda de posição).
- `tools/vista_de_cima.py <ensaio> [--video]`: por quadro gravado, a câmera
  achatada no chão com os cones, as bordas e o centro estimados e a seta do erro
  do Stanley (`<ensaio>/vista_de_cima/`, `vista_de_cima.mp4`).

## Ensaios de 19/11: carro colado na direita (corrigido no mesmo dia)

**Causa:** a calibração de 17/11 foi medida com a trena **girada ~11°** em
relação ao carro, cruzando o eixo dele ~0,9 m à frente. Na correção de 19/11
de manhã, entrou só a metade lateral (origem 18 cm), sem o giro. Perto de 1–2 m,
isso deslocou os cones ~28 cm para a direita, e o carro se "centralizava" colado
nos cones da direita.

**Como o giro foi medido sem recalibrar** (`tools/alinhamento_camera.py`): com
o carro andando, um cone parado se desloca para trás exatamente ao longo do
eixo do carro. Nos ~2.700 pares de cones dos ensaios de 17/11, ele se deslocava
em diagonal: **−11,3°** (cones até 1,4 m e de 1,4 a 2,2 m concordam).
Aplicado com `calibrar_camera_lateral.py --revisar ... --girar 11.3`, o resíduo
fica em +1,4°. Com o giro, a câmera aponta para a frente do carro (0,4°): a
impressão de "câmera virada para a direita" vinha da trena.

O mesmo ajuste mediu uma **guinada que a odometria não vê**: a roda vira ~3,5°
mais para a direita que o comando. O trim aprendido em 17/11 saiu com o sinal
errado (−2°), porque a percepção girada também desloca o esterço médio nas retas.
Agora o trim começa em **+3°** (`steering_trim_deg`), com teto de 6°, num
arquivo novo (`logs/direcao_trim_v2.json`); o antigo não é mais lido.

**Outras correções:**
- **Partida com um cone de cada lado.** Em 19/11 (pista de ~1,9 m), duas voltas
  não saíram do lugar vendo 1 verde e 1 vermelho. Agora o par basta, e o ponto
  de partida do ajuste vem dele: centro no meio, largura entre os dois. Na
  primeira partida, usa o par mais parecido com um portão (mesma distância à
  frente).
- **Trajetória de curva** (até 24/11: `racing_line_m` 0,10, `racing_inside` 1; ver "Curvas fechadas" abaixo;
  `--abrir-curva`, campo "Abrir na curva" na GUI). Com uma curva chegando (pela
  clotoide do estimador, de 0,5 a 1,5 m), a referência vai até 10 cm para
  **fora**: o carro abre antes da curva, vê mais cones dela e entra com folga do
  lado de dentro; dentro da curva volta ao centro. Na pista virtual (1,3 m,
  cones a 1 m, 12 q/s), a folga mínima até os cones foi de 20 para 27–29 cm,
  com todas as voltas completas, também com ruído e erro de calibração. Em pista
  larga (1,9 m) não ajudava (o carro perdia a borda de dentro de vista), então
  o desvio diminui até sumir entre 1,3 e 1,7 m de largura. Fechar para dentro
  na curva (`racing_inside` 1,5) não melhorou na pista estreita e piorou na larga.
- **Visão:** com a fita amarela desligada, foi de 6–7 para **12–15 quadros/s**,
  com 170–190 ms de atraso (era ~280).

**Direção assimétrica** (19/11, depois da correção: curvas à esquerda mais
fechadas que à direita). `steering_left_scale` / `steering_right_scale`
(`--esterco-esquerda`/`--esterco-direita` em %, campos "Esterço à esquerda/à
direita" na GUI) escalam só o comando do servo de cada lado. A odometria
continua com o esterço desejado, que é o que a roda faz com a escala certa.
`tools/alinhamento_camera.py` mede pelos logs quanto da curva pedida a roda faz
de cada lado (separa giro, viés e resposta esquerda/direita) e sugere os dois
valores. Nos logs de 17/11: 89% à esquerda e 104% à direita, mas com outro trim
e só 183 pares em curva à direita.

**Deixe "Debug de cones" ligado.** Os ensaios de 19/11 vieram sem o
`cones_debug.jsonl`, e sem ele não há como medir o giro, a vista de cima nem a
causa de cada batida.

## Ensaios da noite de 19/11 e o que mudou em 24/11

**O que os logs mostraram** (`archive_2411`, voltas de 19/11 a partir das 20 h):
- **Visão:** ~10 quadros/s andando, imagem com 200–255 ms de idade no controle
  (processamento de 16 a 115 ms). O controle roda a 48 Hz.
- **Classificador** desligado na maioria das voltas. Agora vem **ligado por
  padrão na GUI** (rede de 19/11, `modelos/cones.onnx`: 93,6% de acurácia
  balanceada com os ensaios fora do treino).
- **Paradas do ultrassom:** quase todas eram cones de verdade, a 0,05–0,35 m.
- **Trim** aprendido terminou entre +2,7 e +5,6°.
- **Alinhamento** (`tools/alinhamento_camera.py`, calibração atual): giro
  residual de +1,2° (bom). A roda faz **88% da curva pedida à esquerda e 90% à
  direita** (simétrico): a odometria girava ~11% a mais que o carro. Agora
  `steering_response: 0.89` corrige a guinada da odometria (previsão do
  estimador entre quadros e compensação do atraso). O comando do servo não
  muda.
- **Velocidade:** presa em ~1,4 m/s pelo atraso (orçamento de 0,3 m). Agora é
  0,5 m (~2,3 m/s), e o teto vai até 4 m/s (ver "Rápido nas retas").

### Mapa de cones (SLAM) e traçado de margem máxima (SVM)

Ideia dos professores, aplicada em `fva_car/cone_slam.py`,
`fva_car/svm_path.py` e `fva_car/map_path.py`.

**SLAM (EKF-SLAM de cones).** O estado tem a pose do carro e a posição de cada
cone, `[x, y, θ, c1x, c1y, …]`, no referencial do mapa.
- **Previsão:** a odometria do eixo dianteiro. O ruído cresce com o caminho
  andado.
- **Medida:** cada cone do quadro, com a covariância da homografia (pequena na
  direção, grande na distância). Ele é associado ao cone de mesma cor mais
  próximo pela distância de Mahalanobis, com portão de 99% (χ² = 9,21).
- **Rever um cone corrige a pose do carro**, ou seja, a deriva da odometria.
- **Limpeza do mapa:** um cone visto uma vez só e não revisto em 1 s sai
  (reflexo, falso positivo). Cones mais de 1,2 m atrás do eixo também saem
  (até 40 no mapa).

O mapa guarda os cones **ao lado e atrás** do carro, que a câmera (0,8–3 m à
frente) já não vê. No simulador, os cones do mapa ficam a 3–4 cm (mediana) dos
reais.

**SVM.** A borda esquerda é uma classe (+1), a direita a outra (−1). O SVM de
kernel RBF acha a fronteira de **margem máxima**, isto é, o caminho mais
longe dos cones dos dois lados. Detalhes:
- O SVM é só numpy: o dual com o viés no kernel é resolvido por gradiente
  projetado. O Raspberry não tem scikit-learn.
- As bordas são preenchidas entre cones vizinhos a cada 25 cm. Com cones a 1 m,
  o SVM só com os cones fecha a fronteira em volta de cada um.
- Com uma borda só, a outra é virtual, a uma largura de distância.
- Um polinômio que pondera os pontos perto dá (e, rumo, curvatura) no eixo.
  O traçado só vale se cobrir o eixo: extrapolar 1,5 m dava rumos de 60°.

**Como entra no carro: reserva do estimador** (`cone_map: true`, padrão; caixa
"Mapa SLAM + traçado SVM" na GUI; `--sem-mapa` desliga):
- O SLAM roda a cada quadro, ~0,8 ms no notebook.
- O SVM só roda quando o estimador **perde** o corredor, ~6 ms.
- Nesse caso, se o mapa tem cones dos dois lados a menos de 2 m, o carro segue
  devagar pelo traçado do mapa (estado `MAPA`) em vez de parar.
- O `cones_debug.jsonl` ganha `mapa` com o número de cones no mapa, quantos de
  cada lado e se o traçado foi usado.

**Por que não no lugar do estimador** (`--mapa-substitui`, só para comparar).
Na pista virtual, 5 pistas × 4 sementes:

| cenário | estimador (rms / folga mín.) | mapa no lugar | mapa de reserva |
|---|---|---|---|
| 12 q/s, 1,3 m | 6,7 cm / 29 cm | 9,0 cm / 7 cm | 6,7 cm / 29 cm |
| 12 q/s, 1,9 m | 6,6 cm / 41 cm | 13,1 cm / 5 cm | 6,6 cm / 41 cm |
| ruído + calibração, 1,3 m | 7,0 cm / 27 cm | 9,9 cm / 12 cm | 7,0 cm / 27 cm |

Com cones a 1 m, a corda entre dois cones de fora **corta o arco** da curva, e o
centro de margem máxima fica 6–15 cm para dentro numa curva fechada. O
estimador já usa um modelo de pista (clotoide com largura conhecida) e erra
5–6 cm. Como reserva, nos cenários difíceis (4–7 q/s, iscas, ruído; 72 voltas
por caso), o mapa ficou empatado com o estimador sozinho (±2 voltas). No carro
real, comparem com e sem a caixa: o `jsonl` mostra quando o mapa assumiu.

### SVM como classificador de cones (comparação)

O mesmo conjunto da rede (6.400 recortes 32×32 rotulados, validação com
ensaios inteiros fora do treino) foi testado com **HOG + histograma de cor +
SVM**:

| classificador | acurácia balanceada | rejeita não-cone | verde | vermelho |
|---|---|---|---|---|
| **rede (CNN, 23.891 parâmetros)** | **93,6%** | **87,7%** | 96,3% | 96,9% |
| HOG + SVM linear | 89,5% | 78,5% | 94,0% | 96,0% |
| HOG + SVM RBF (4.000 features aleatórias) | 90,4% | 79,0% | 95,8% | 96,4% |

A rede ganha principalmente em rejeitar o que não é cone (batente, reflexo).
Ela continua sendo a do carro.

## IMU: guinada pelo giroscópio (24/11)

Sem IMU, a guinada do carro vinha do modelo da bicicleta, `v·tan(esterço)/L`,
com o esterço **comandado**. Nos logs de 19/11 a roda fazia só 88–90% da curva
pedida, e o pneu escorrega mais quanto mais rápido. A guinada errada entrava na
previsão do estimador entre quadros, na compensação dos ~220 ms de atraso e no
mapa (SLAM).

**Como funciona** (`fva_car/yaw_rate.py`, driver `fva_car/imu.py`):
- **Partida, com o carro parado (~1 s):**
  - o chip precisa se identificar (WHO_AM_I: MPU-9250/9255/6500/6050);
  - a gravidade dá o eixo vertical, então a IMU pode estar montada em qualquer
    posição (de lado, de cabeça para baixo) sem mudar sinal no código;
  - o giroscópio dá o viés;
  - o giroscópio é configurado em ±500°/s (3 m/s numa curva de 1 m dão ~170°/s),
    com filtro interno de 41 Hz e 100 Hz de amostragem;
  - a leitura é um bloco I2C de 14 bytes, não 12 transações.
- **Na volta:** a odometria usa a guinada medida, com média de 2 amostras em vez
  das 10 do modelo, o que tira ~100 ms de atraso da orientação. Parado, o viés é
  reaprendido devagar (deriva com a temperatura).
- **Resposta da direção:** em curva, a razão guinada medida / guinada do comando
  é medida para cada lado. O terminal mostra isso no fim da volta e sugere
  `--esterco-esquerda/--esterco-direita` se os lados forem diferentes.

**A IMU já queimou uma vez: ela nunca é obrigatória.** Em qualquer falha, o
`Car` usa o modelo da direção (com `steering_response` 0,89), como antes:

| situação | o que acontece |
|---|---|
| sem `smbus`, chip ausente no I2C, WHO_AM_I errado (queimado) | partida sem IMU, aviso amarelo |
| parado, acelerômetro longe de 1 g, leituras todas iguais, viés > 20°/s, carro mexendo | partida sem IMU, com o motivo |
| erro de I2C isolado na volta | aquele passo usa o modelo |
| 10 erros seguidos, leituras idênticas por 1 s (travada) | IMU desligada até o fim da volta |
| leitura saturada | aquele passo usa o modelo |
| em curva, guinada medida < 30% da do comando (eixo morto) | IMU desligada até o fim da volta |

O resultado fica em `imu.json`, no `cones_summary.json`, na linha `imu:` do
relatório do ensaio e na coluna `yaw_from_imu` do `car.csv` (1 = giroscópio).
`--sem-imu` (caixa "IMU: guinada pelo giroscópio" na GUI) volta ao modelo, para
comparar.

**Pista virtual** (carro que faz 80–89% da curva pedida, viés de 2° na direção,
10 q/s com 220 ms de atraso, 5 pistas × 10 sementes):

| cenário | modelo (0,89) | giroscópio |
|---|---|---|
| 0,7 m/s, 1,3 m | 7,6 cm rms, 1 toque | 6,2 cm, 0 toques |
| teto 2,8 m/s, 1,3 m | 8,7 cm, 3 toques, 93% das voltas | 6,4 cm, 0 toques, 100% |

**Amortecimento pela guinada** (`stanley_yaw_damping`, `--amortecer-guinada K`).
O Stanley ganha o termo `−k·(guinada medida − v·curvatura do caminho)`
([Stanley em corrida de carros RC](https://arxiv.org/pdf/2408.15152)): o
giroscópio vê o carro começando a oscilar na hora, enquanto a imagem chega
~220 ms depois. Na pista virtual, com k de 0,1 a 0,5, o erro médio caiu pouco
(6,5 → 6,3 cm), mas os toques variaram para os dois lados. Fica **desligado**
(0). Para testar no carro: `--amortecer-guinada 0.2`, comparando com uma volta
sem. Sem IMU válida, o termo é zero.

## Ensaios de 24/11 à noite: curvas, rotatória e controle da missão

### Controle da missão pela GUI (aba 📷 Câmera)

Botões no quadro "Missão" da aba da câmera, junto do vídeo:
- **▶ Iniciar:** roda o comando da aba Comandos.
- **⏹ Encerrar:** para o carro e salva os logs.
- **🔄 Reiniciar:** encerra, espera o fim e inicia de novo.
- **🔓 Liberar parada:** destrava a parada do ultrassom ou da placa de PARE.

A resposta do carro aparece ao lado dos botões e no terminal da aba Comandos.

**Tempo da missão:** 50 s, no `main.py` e no campo "Tempo da missão (s)" da GUI
(`--tempo` muda).

**Como chega ao carro** (24/11, depois de os botões "não funcionarem"):
1. A GUI escreve `<ação> <código>` em `/tmp/fva_missao.cmd`.
2. O laço de controle lê esse arquivo a cada 0,1 s e responde em
   `/tmp/fva_missao.ack`.
3. Na GUI aparece "✅ … recebido pelo carro".
4. Sem resposta em 3 s, a GUI manda o sinal como reserva (`SIGUSR1`/`SIGTERM`),
   mas **só se o processo trata esse sinal**. Num `main.py` antigo, o `SIGUSR1`
   mataria o programa sem parar os motores direito; nesse caso a GUI recusa e
   diz "é o código antigo; envie os arquivos para o carro".
5. A missão é achada pelo arquivo `/tmp/fva_missao.pid`. Sem ele (o `main.py`
   antigo não o grava), a GUI procura o `main.py` entre os processos do carro.
   Se não achar nenhum, diz o motivo ("nenhum main.py rodando…").

O código antigo no carro não tinha o canal, então **enviem os arquivos** (aba
Enviar Arquivos) antes de testar.

**Liberar com o cone ainda na frente:** o carro quase sempre para encostado no
cone. Antes, ao liberar, o ultrassom via o mesmo cone e travava de novo em
0,15 s: o bipe não parava e parecia que o botão não funcionava. Agora, depois de
liberar, o ultrassom fica ignorado por `obstacle_resume_grace_s` (2 s), para o
carro sair de perto. Depois disso, um obstáculo de verdade para o carro de novo.

### Curvas fechadas: chegar por fora, convergir para o ápice

Nos logs de 24/11 o carro passava a curva 5–9 cm **por dentro** do alvo e
encostava nos cones de dentro. Em 55–63% do tempo ele via só a borda de fora.
Três causas:
- **O traçado antigo** abria só 4–6 cm antes da curva e, na saída, fechava para
  dentro.
- **A traseira** passa por dentro da dianteira (L²κ/2, ~5 cm numa curva de
  R = 1,25 m), e o Stanley controla só o eixo dianteiro.
- **A largura da pista** era medida pelo primeiro par de cones: 1,10 m numa volta
  e 1,88 m na seguinte, na mesma pista. Vendo só a borda de fora, com 1,68 m o
  centro vai ~20 cm para dentro.

**Traçado "ápice"** (`racing_mode: "apice"`, padrão):
- **Antes da curva:** a referência vai até `racing_line_m` (0,15 m) para
  **fora**. O carro chega aberto, faz a curva com raio maior e vê mais cones
  dela.
- **Dentro da curva:** converge para o ápice, até `racing_apex_m` para dentro,
  limitado pela folga aos cones de dentro e com o arrasto da traseira
  compensado.
- **Na saída:** com a curvatura caindo, volta ao centro.

Pista virtual (curvas fechadas, 85% de resposta da direção, 220 ms de atraso,
16 sementes). A folga é medida na **carroceria inteira**, do eixo traseiro à
frente, não só no eixo dianteiro:

| pista | folga mediana aos cones de dentro: antes → agora | voltas completas |
|---|---|---|
| 1,3 m | 31,5 → **35,8 cm** | 98% → **100%** |
| 1,6 m | 46,9 → **50,4 cm** | 97% → 98% |

Levar o carro de fato para dentro no ápice (0,05 m, com 0,25 m para fora) tirou
folga (25 cm). Por isso o ápice fica em 0 por padrão. `--apice 0.05` testa, e
`--abrir-curva` (campo "Abrir na curva" na GUI) muda a abertura.

**Largura da pista medida** (`--largura-pista 1.3`, campo "Largura da pista" na
GUI): o estimador já começa sabendo a largura e a aprendida não sai de ±15%
dela. Meçam com a trena de cone a cone. Vazio, o carro estima como antes.

### Rotatória: cores e engasgo

Pista virtual com rotatórias de rua (entrada virando à direita, volta inteira
na ilha, saída) e a **mesma decisão de cor do carro**:
- **Antes:** o carro "trocava as cores" 29 vezes em 36 voltas. De dentro do anel,
  ele via pelas aberturas os cones verdes das vias de entrada e saída do lado de
  fora (à direita), concluía que o verde era a borda direita e, a cada troca,
  recomeçava o estimador do zero. Ficava "perdido" e parava: o engasgo.
- **Agora:**
  - só votam cones na faixa de uma borda (20–85% da largura até o centro);
  - uma troca decidida pelo corredor mantém a geometria (só a cor de cada borda
    muda) e não recomeça mais;
  - uma troca na partida, sem corredor, ainda recomeça, porque ali o estado foi
    montado com as cores trocadas.

| 36 voltas em rotatórias | antes | agora | cores ideais |
|---|---|---|---|
| trocas de cor | 29 | **0** | 0 |
| paradas | 23 | **8** | 8 |
| percurso feito | 39% | **51%** | 51% |

Nas pistas normais, nada mudou. **O que ainda falta:**
- **Sair da rotatória:** é uma decisão. De dentro do anel, continuar girando e
  pegar a saída são os dois caminhos válidos, e o carro não tem regra para
  escolher; as voltas que falham no simulador falham ali.
- **Ilha de cor trocada:** se a ilha tiver a cor oposta à da borda esquerda da
  entrada, o carro não percebe a troca, porque dentro do anel vê quase só a borda
  de fora.

### Esterço "pedalando": memória do traçado

Nos 5 ensaios das 19h25–19h30 o esterço trocou de lado rápido (mais de 8° de um
lado ao outro em menos de 0,5 s) de 5 a 7 vezes por volta, às vezes em
sequência: −8 → +10 → −8 → +10° em menos de 1 s. A causa: entre quadros, o
seguidor leva o traçado pela odometria (com o giroscópio), mas cada quadro novo
discordava do levado em ~0,15 rad de rumo. Com uma borda só e 2–4 cones, o
ajuste oscila de quadro a quadro, e esse salto fica abaixo do limite de descarte
(0,35 rad), então o Stanley obedecia cada um.

**Memória do traçado** (`path_memory_s` 0,25; `fva_car/lateral.py`
`_remember_path`):
- Num quadro novo, a diferença para o traçado já identificado, levado até agora,
  vira um resíduo que some com constante de tempo de 0,25 s.
- Um traçado novo de verdade vale em ~0,75 s.
- Um quadro ruim isolado mal mexe no esterço.

Pista virtual (ruído do carro real, 10 q/s, 220–300 ms, 12 sementes):

| cenário | pedaladas: sem → com | voltas | toques |
|---|---|---|---|
| 1,3 m | 48 → **13** | 100% → 100% | 0 → 1 |
| 1,5 m, 0,8 m/s | 46 → **22** | 100% → 100% | 1 → 1 |
| 1,3 m + 6 objetos falsos ao lado | 130 → **69** | 76% → 72% | 19 → 21 |

**Custo:** com objetos falsos, a correção de um salto do estimador também
demora, por isso o caso com objetos falsos piorou um pouco. Exigir dois quadros
concordando antes de mudar foi pior (pedaladas 48 → 39 só). `--memoria-tracado
0` ou a caixa "Memória do traçado" na GUI desligam, para comparar.

### Placas ArUco (PARE)

`fva_car/aruco.py`, dicionário 5×5 (IDs 0–99), o mesmo do `camera.py`.
- **Busca:** a cada 2 quadros na imagem cheia (~1,3 ms no notebook).
- **No vídeo:** as placas aparecem com um quadrado magenta e o ID.
- **Ação por ID:** `aruco.actions` no `main.py`, decidida num `match`/`case`.
  Hoje só a ID 1 tem ação (`"pare"`); `"ignorar"` só registra.

**PARE (ID 1), placa AO LADO da pista:**
1. A placa precisa ser vista em 2 quadros seguidos, para um reflexo não parar o
   carro. O carro então reduz para 0,3 m/s (`approach_speed`).
2. **Onde está a placa** (`fva_car/camera_pose.py`):
   - a calibração guarda só a homografia do chão; dela sai o foco da câmera
     (521 px, 63° de campo) e a altura dela (~25 cm);
   - a **direção** da placa vem da própria homografia, e a **distância** vem do
     tamanho aparente do marcador (o lado vertical, que não encolhe com a placa
     vista de lado);
   - o terminal mostra a distância, o lado e a **altura estimada do centro da
     placa**. Se essa altura não bater com a real, o tamanho configurado está
     errado.
3. **Lembrada pela odometria:** a placa ao lado sai da imagem ~1,5 m antes de o
   carro chegar. A posição dela fica guardada no referencial da odometria, com
   peso maior para as leituras de perto, e o carro segue contando quanto falta.
4. **Parada:** quando falta só a distância de frear, o carro para, com a
   **frente do carro na altura da placa** (`stop_at_m` 0; negativo para antes).
   Se a placa estiver na frente, o ultrassom também para (8 cm do pneu ao cone).
5. Fica parado até **🔓 Liberar parada** (ou `--pare-segundos S`) e depois
   ignora a placa por 4 s. Se a placa sumir sem posição e sem parada, é
   esquecida em 2,5 s.

**Medir antes do ensaio:** o lado do **quadrado preto** do marcador impresso
(`marker_m`, padrão 0,10 m; `--placa-lado 0.12`). Um erro de 20% no tamanho
desloca a parada ~20% da distância da última leitura (~30 cm a 1,5 m). No teste
com 5% de ruído no tamanho, a frente do carro parou a menos de 12 cm da placa.

O terminal mostra cada passo; o `car.csv` tem `aruco_state` (0 livre,
1 aproximando, 2 parado, 3 saindo), `aruco_id` e `aruco_remaining_m` (quanto
faltava até a placa); o `cones_debug.jsonl` tem as
placas de cada quadro. `--sem-aruco` ou a caixa "Placas ArUco" na GUI desligam.

## Relatório do ensaio e versão do código

- `main.py` imprime `Codigo: versao xxxxxxxx` (hash dos `.py`) e grava esse
  valor em `cones_summary.json` e `vision_status.json`. A GUI compara as
  versões do notebook e do carro antes de rodar e avisa se forem diferentes.
- `python3 tools/analisar_ensaio.py <pasta> [--linha]`, ou o botão
  **📋 Relatório** na aba de dados, traz:
  - a versão do código;
  - quadros por segundo e idade da imagem (metas: 15 q/s e 150 ms);
  - as paradas do ultrassom (eco isolado ou leituras seguidas);
  - o erro lateral e o rumo RMS (metas: 5 cm e 6°);
  - o centro estimado em reta e nas curvas, as trocas de lado e o trim;
  - as opções da volta (classificador, refino da base, partida nova ou
    antiga, acelerador mínimo) e a **partida**: tempo até 80% da velocidade
    de cruzeiro (meta 1,5 s), tempo preso a 0,35 m/s, quantas vezes o motor
    parou nos primeiros 5 s e a maior esterçada com o carro parado.

  Com `--linha` e quadros gravados, mede também a posição real do carro pela
  fita e o desvio da calibração (centro dos cones − fita a 1 m). O relatório é
  salvo em `<pasta>/relatorio.txt`.

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
2. **Rotular "é cone? sim / não"** no notebook (`tools/rotular_sim_nao.py`).
   O jeito mais rápido é o botão **🏷 Rotular cones (sim/não)** na aba de
   dados da GUI. Ele recorta os candidatos de **todos** os quadros coletados (só
   os novos, a cada vez: os rótulos feitos ficam) e abre a grade maximizada,
   com os recortes ocupando a tela toda (colunas e linhas pelo tamanho da tela;
   F11 = tela cheia), já pré-marcados pelo detector atual (borda verde = sim,
   vermelha = não). Você só clica nos errados e aperta Enter. O botão direito
   mostra o quadro inteiro, e `t` marca a página toda como "não". Pela linha de
   comando (`--colunas`/`--linhas` fixam a grade):
   ```bash
   python3 veiculo_real/tools/rotular_sim_nao.py preparar ~/fva_experimentos --saida ~/fva_experimentos/rotulagem
   python3 veiculo_real/tools/rotular_sim_nao.py rotular ~/fva_experimentos/rotulagem
   python3 veiculo_real/tools/rotular_sim_nao.py exportar ~/fva_experimentos/rotulagem --out ~/fva_experimentos/rotulagem/cones.npz
   ```
   Usa 1 quadro a cada 2 (`--passo`), porque quadros vizinhos repetem os
   mesmos cones, e no máximo 15 candidatos por quadro. Precisa de quadros
   gravados (`frames/`, opção "Gravar quadros" da GUI). A foto única
   `vision_camera.jpg` entra também, mas rende pouco. Um cone que nem a cor
   larga pegou não aparece na grade; para esses, use o modo antigo
   (`rotular_cones.py`, clicar na base de cada cone) e o
   `extrair_recortes_cones.py`.
3. Treinar no notebook (`pip install torch --index-url https://download.pytorch.org/whl/cpu onnx`):
   `python3 veiculo_real/tools/treinar_classificador_cones.py ~/fva_experimentos/rotulagem/cones.npz --out veiculo_real/modelos/cones.onnx`
   (aceita vários `.npz`, inclusive os do modo antigo).
   A validação é separada por quadro. O script mostra a matriz de confusão e
   recarrega o modelo no OpenCV para confirmar que a saída bate com o PyTorch.
4. Medir antes e depois:
   `python3 veiculo_real/tools/benchmark_visao.py --cones --image logs/<ensaio>/vision_camera.jpg --classifier veiculo_real/modelos/cones.onnx`
   e `avaliar_cones.py` nos quadros rotulados que ficaram fora do treino.
5. Usar no carro: na GUI, marque **Classificador (rede)** na aba da missão (o
   caminho padrão já é `modelos/cones.onnx`, que vai junto na sincronização).
   Sem a GUI: `"cone_classifier": "veiculo_real/modelos/cones.onnx"` em
   `main.py`, ou `--classifier ...` na linha de comando (`--classifier none`
   desliga).

**Primeiro modelo (17/11).** 3.779 recortes de 10/11 rotulados sim/não: 405
não-cones, 1.538 cones verdes e 1.836 vermelhos. Os 511 palpites corrigidos
eram, na maioria, batentes de porta e objetos vermelhos que o detector aceitava.
- **Validação por quadro:** revocação de 0,82 (não-cone), 0,97 (verde) e 0,97
  (vermelho).
- **Ensaios inteiros fora do treino** (3 sorteios de 2 ensaios), a medida
  honesta: 97-100% dos cones mantidos e 59-82% dos falsos descartados.

Tudo vem de um só lugar e dia; dados de outro ambiente vão melhorar o modelo.

O `.json` ao lado do `.onnx` guarda as classes, o recorte, os ajustes de cor
dos candidatos usados no treino (aplicados automaticamente, salvo se
`cone_settings` disser outra coisa) e a confiança mínima (`min_confidence`,
0,5 no modelo atual; o treino grava 0,5 por padrão). O debug por quadro registra os candidatos, os aceitos e o tempo
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
