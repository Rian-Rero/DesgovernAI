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

A calibracao da camera do carrinho nao foi encontrada no projeto. O exemplo
`calibracao/camera_lateral.example.json` tem listas vazias intencionalmente:
nao representa medidas reais e nao habilita movimento. A missao valida um
arquivo de calibracao antes de criar `Car` ou abrir os atuadores.

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

A referencia inicial e 0.5 m/s, maximo de 20 graus e taxa de 60 graus/s.
Ganhos, sinais e limites ainda precisam de ajuste e verificacao no carro
fisico. O entre-eixos do `Car` real continua 0.36 m; nao foi substituido pelo
valor de 0.302 m do simulador. `steering_sign=+1` segue a convencao do codigo
real (positivo para esquerda) e pode ser invertido se a montagem do servo
exigir. A calibracao de trim existente nos servos e preservada.

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
frame antigo por mais de 0.3 s, falha na thread ou emergencia cortam throttle
e zeram o integral do PI. A parada por obstaculo ja existente permanece
travada ate o fim da missao. Cortar throttle coloca o ESC em neutro; isso
nao equivale a uma frenagem ativa e o carrinho ainda percorre uma distancia.
A atuacao fica na thread principal; a visao apenas publica observacoes.

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
