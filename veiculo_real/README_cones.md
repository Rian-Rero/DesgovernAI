# Centro dos cones no carrinho real

O modo `--navigation cones` prioriza o corredor de cones e, se ele não for
válido, procura a linha amarela na mesma imagem. Se nenhuma referência for
válida, neutraliza os comandos. Quando os cones voltam, recebem prioridade.

A GUI `tools/fva_gui_um_carro.py` já inicia nesse modo com `--stream 8080`.
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

A transmissão mostra as bases vermelhas/verdes e o centro calculado em ciano.
`--display` exibe o resultado localmente. O modo padrão continua sendo a linha
amarela; `--navigation line` o seleciona explicitamente.

## Geometria e controle

`fva_car/cones.py` segmenta vermelho/laranja e verde em HSV OpenCV. Projeta as
bases visíveis para metros pela homografia de `camera_lateral.json`, cuja
origem deve ser o centro do eixo dianteiro, com +x para frente e +y à esquerda.
A câmera deve permanecer na posição usada na calibração.

O detector precisa de pelo menos dois cones em cada borda, com extensão
longitudinal de 30 cm e cobertura comum de 30 cm entre as bordas. Dois cones
definem uma reta; três ou mais permitem estimar uma curva quadrática. As duas
bordas são avaliadas nas mesmas distâncias à frente: a média forma o centro.
Não se cria uma borda ausente nem um mapa completo a partir de uma imagem.
Bordas cruzadas, cobertura apenas distante e largura incoerente invalidam o quadro.

O Stanley já existente recebe erro lateral, orientação e curvatura desse centro.
Mantém antecipação pela curvatura, redução de velocidade em curvas, compensação
de atraso, sinais/limites do servo e proteções de encoder, RC, emergência e
obstáculos. No modo de cones, perder o corredor aciona a linha amarela. Se ambas falharem,
neutraliza os comandos no primeiro quadro inválido. A busca sem observação
fica desativada. A troca de referência limpa a memória da referência anterior.

## Calibração e ajustes

A calibração atual contém medidas a 0,30–0,60 m do eixo dianteiro. Ela não
comprova precisão em toda a área necessária para ver os cones. Confira o
campo de visão e calibre com pontos no piso cobrindo as distâncias e larguras
usadas. Se mudar a inclinação da câmera para enxergar mais longe ou as duas
bordas, refaça a calibração com `tools/calibrar_camera_lateral.py`.
A homografia vale para o contato no piso, não para o topo dos cones.

Em `main.py`, `cone_settings` permite ajustar:

- `red_hsv_ranges`: pares `[mínimo, máximo]` de HSV (padrão H 0–20 e 170–179).
- `green_hsv_ranges`: pares HSV (padrão H 35–85).
- `min_area_px`: área mínima na resolução calibrada (padrão 12 pixels).
- `min_width_m` e `max_width_m`: largura normal admissível (padrão 0,5–5,5 m).

H usa 0–179; S e V usam 0–255. Ajuste a segmentação conforme a iluminação.
A detecção por cor pode confundir objetos vermelhos/verdes com cones; esta
implementação ainda precisa ser validada com imagens e ensaios do veículo real.
Cones cortados nas bordas da imagem são descartados por não fornecerem uma base
confiável. Cones com faixa branca devem manter a parte colorida inferior visível.

Os logs acrescentam `cone_red_count`, `cone_green_count` e `corridor_width_m`.
`line_valid`, `cross_track_m`, `heading_error_rad` e `curvature_m_inv` continuam
representando a referência usada pelo controlador, agora o centro do corredor.
