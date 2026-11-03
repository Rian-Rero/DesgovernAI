# Cones como referencia principal do Stanley

O controle do simulador escolhe uma unica referencia visual por passo:

1. **Cones vermelhos e verdes:** reconstrucao do centro do corredor.
2. **Linha amarela:** alternativa quando o corredor nao pode ser estimado com confianca.
3. **Sem referencia valida:** velocidade de referencia zero e frenagem pelo PI longitudinal.

As duas referencias alimentam o mesmo controlador Stanley. A troca de fonte conserva
o limite de velocidade do esterco, evitando um salto de comando. O veiculo real
continua com sua implementacao anterior.

## Executar

Abra no CoppeliaSim a cena `simulador/coppeliasim/simulador_cones_sem_linha.ttt`
e execute, a partir da raiz do repositorio:

```bash
venv/bin/python simulador/main.py --duration 240 --speed 1.5
```

A cena `simulador_cones.ttt` permite exercitar a alternativa da linha amarela.
O programa conecta ao CoppeliaSim ja aberto, prepara a simulacao e executa em passos
sincronizados. `--host`/`--port` selecionam a API (padrao `127.0.0.1:23000`); `--no-display` desliga
os graficos; `--no-real-time` remove a espera para acompanhar o relogio real.

As dependencias estao em `simulador/requirements.txt`.

## Da imagem ao comando

A entrada e uma imagem **RGB** do sensor, corrigida verticalmente pelo `Car`.
A segmentacao por cor identifica regioes vermelhas e verdes e encontra a base
visivel de cada cone. A base e mais adequada que o centro da mancha porque
corresponde aproximadamente ao contato com o chao.

A perspectiva e a pose da camera convertem cada base de pixels para metros,
intersectando o raio visual com o plano da pista. O controle usa apenas essas
observacoes da camera: nao le o nome, o identificador ou a posicao dos objetos
cones da cena.

Os pontos das duas cores definem as bordas do corredor. A estimativa central
compara as bordas na mesma distancia longitudinal, em vez de juntar simplesmente
o cone vermelho mais proximo com o verde mais proximo. Isso permite que os cones
estejam desencontrados e que o corredor faca curvas. Uma memoria local limitada
preserva pontos ja vistos quando os cones proximos saem do campo da camera.
As observacoes repetidas sao associadas espacialmente por cor para manter um ponto
filtrado por cone. Um ajuste robusto separa as bordas dos cones de outros trechos
que aparecem ao fundo. Curvas com apoio suficiente tambem podem ser modeladas como
arcos concentricos, cujo raio central fica entre os raios das duas bordas.

Os ajustes ficam no dicionario `parameters` passado a `run_mission`:

| Parametro | Padrao | Funcao |
| --- | --- | --- |
| `cone_max_distance` | 12 m | Alcance longitudinal usado no corredor |
| `cone_memory_horizon` | 6 s | Idade maxima dos pontos de apoio ja vistos |
| `cone_freshness` | 0,5 s | Tempo maximo sem deteccao nova de cada cor |
| `cone_min_span` | 1 m | Comprimento minimo de apoio por borda |
| `cone_max_near_distance` | 6 m | Distancia maxima do apoio mais proximo |
| `cone_min_width` / `cone_max_width` | 0,75 / 7 m | Intervalo de largura aceitavel |
| `cone_ground_z` | 0,005 m | Altura do plano de contato no mundo |

E necessario apoio de pelo menos dois cones de cada cor, comprimento e
sobreposicao longitudinal suficientes e bordas sem cruzamento. A estimativa
curva requer mais apoio que uma reta. A memoria conserva a geometria proxima,
mas cada cor precisa continuar sendo vista dentro do intervalo de frescor.

O caminho central fornece tres medidas no eixo dianteiro:

- `cross_track_m`: distancia lateral assinada ate a trajetoria; positiva a esquerda.
- `heading_error_rad`: diferenca entre a tangente da trajetoria e a direcao do carro.
- `curvature_m_inv`: curvatura local; positiva em curvas para a esquerda.

O Stanley calcula o esterco geometrico:

```text
esterco = erro_angular + atan2(ganho * erro_lateral, |velocidade| + suavizacao)
```

O comando respeita o limite de 20 graus e o limite de variacao por segundo. O
adaptador inverte o sinal para as juntas da cena, cuja convencao positiva vira
para a direita. A referencia longitudinal diminui com a curvatura e com os erros
de seguimento. O PI existente controla a velocidade resultante.

Quando ambas as referencias sao perdidas, o carro freia. Durante no maximo 0,4 s,
o esterco anterior pode ser mantido enquanto ele desacelera; isso evita desfazer
uma curva no meio da frenagem.

## Visualizacao e registro

A imagem mostra as bases detectadas, o caminho central e a fonte selecionada.
As distancias vermelha e verde sao laterais assinadas em metros, relativas ao
eixo dianteiro: esquerda positiva e direita negativa. A igualdade de seus modulos
indica centralizacao em um trecho aproximadamente reto.

Os graficos mostram os erros do caminho selecionado, a velocidade de referencia,
a velocidade medida e o esterco. O CSV `logs/<experimento>/car.csv` inclui:

| Coluna | Significado |
| --- | --- |
| `path_source` | 2 = cones; 1 = linha amarela; 0 = sem referencia |
| `path_valid` | Validade da referencia efetivamente utilizada |
| `cones_valid` | Validade do corredor reconstruido |
| `line_valid` | Linha amarela selecionada e valida |
| `yellow_evaluated` | Segmentacao amarela executada neste passo |
| `red_cones`, `green_cones` | Quantidade de bases detectadas na imagem atual |
| `red_lateral_m`, `green_lateral_m` | Distancia lateral de cada borda |
| `cross_track_m`, `heading_error_rad` | Erros enviados ao Stanley |
| `curvature_m_inv` | Curvatura da trajetoria selecionada |

`line_valid=0` com `yellow_evaluated=0` significa que os cones tiveram prioridade;
nao significa que a faixa amarela desapareceu.

## Limites da estimativa

A medida monocular supoe pista plana e a calibracao da camera do simulador.
Objetos coloridos parecidos com cones e corredores muito ocluidos exigem os filtros
de consistencia geometrica; uma deteccao isolada nao basta para liberar o seguimento.
Os pontos antigos expiram: a memoria nao permite navegar indefinidamente sem visao.

Para executar as verificacoes automaticas:

```bash
MPLBACKEND=Agg venv/bin/python -m unittest discover -s simulador/tests -v
```
