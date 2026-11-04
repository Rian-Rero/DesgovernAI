# Centro dos cones no carrinho real

O modo padrão `--navigation cones` usa somente cones. Não há alternância
para linha amarela. Se o detector não fornecer uma referência métrica válida
de cones, o controle neutraliza os comandos. O modo de linha continua disponível
apenas por seleção explícita com `--navigation line`.

A GUI `tools/fva_gui_um_carro.py` já inicia nesse modo com `--stream 8080`.
O comando da GUI usa `--vision-width 0` para preservar a resolução calibrada
e os detalhes dos cones próximos ao horizonte.
O seletor mantém as velocidades 0,7, 1,0 e 1,5 m/s; o valor inicial é 0,7 m/s.
O exemplo abaixo usa 0,3 m/s pela linha de comando.

O detector de cones usa a câmera BGR e a calibração do piso para estimar as
posições das bases dos cones. Vermelhos e verdes podem trocar de lado:
a geometria identifica qual borda fica à esquerda em cada imagem.
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
largura, margem e raio mínimo. O rastreamento e a memória são os mesmos da
interface: cones persistem por até 350 ms; uma borda visível pode usar a largura
aprendida por até 1500 ms, sujeita ao limite de confiança. `tracking=False`
desativa essa estimativa. Quando não resta uma referência de cones válida,
o controle neutraliza os comandos. O adaptador calcula os erros do
Stanley junto ao eixo dianteiro a partir do centro métrico.

O Stanley já existente recebe erro lateral, orientação e curvatura desse centro.
Mantém antecipação pela curvatura, redução de velocidade em curvas, compensação
de atraso, sinais/limites do servo e proteções de encoder, RC, emergência e
obstáculos. No modo de cones, uma referência inválida neutraliza os comandos no primeiro
quadro. A busca de linha amarela e o chaveamento automático ficam desativados.

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
- `min_area`: área mínima na escala 640×480 (28 pixels).
- `min_span` / `max_gap`: extensão e continuidade das bordas na imagem.
- `min_width_cm` / `max_width_cm`: largura admissível (40–500 cm).
- `tracking`, `memory_ms`, `width_memory_ms`: rastreamento e memória.

Os campos são os de `ConeSettings` da interface HSV; a identificação dos lados
continua automática, mesmo que `green_left` indique a preferência inicial.

H usa 0–179; S e V usam 0–255. Ajuste a segmentação conforme a iluminação.
A detecção por cor pode confundir objetos vermelhos/verdes com cones; esta
implementação ainda precisa ser validada com imagens e ensaios do veículo real.
Cones cortados nas bordas da imagem são descartados por não fornecerem uma base
confiável. Cones com faixa branca devem manter a parte colorida inferior visível.

Os logs acrescentam `cone_red_count`, `cone_green_count` e `corridor_width_m`.
`line_valid`, `cross_track_m`, `heading_error_rad` e `curvature_m_inv` continuam
representando a referência usada pelo controlador, agora o centro do corredor.

A transmissão mostra a contagem de cones aceitos por cor, inclusive quando
a geometria do corredor é inválida. Regiões unidas por sobreposição podem
fornecer a base inferior visível; bases ocultas não são inventadas.
