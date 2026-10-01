# ADAS de frenagem frontal

O `main.py` chama `car.set_adas_vel(MAIN_VEL, dist, valid)` em cada ciclo.
O ADAS usa a maior velocidade entre encoder bruto e filtrado para nao
subestimar a distancia de parada durante a aceleracao.

## Decisao

Com velocidade `v`, desaceleracao de coast-down `a_coast`, margem `d_min`
e atraso `tau`, a distancia estimada para parar em roda livre e:

```text
d_coast = d_min + v * tau + v^2 / (2 * a_coast)
d_ativacao = d_coast + v * anticipation_time
```

- `CRUISE (0)`: distancia acima de `d_ativacao`; usa o PI de velocidade.
- `COAST (1)`: distancia entre `d_coast` e `d_ativacao`; neutro e bip.
- `BRAKE (2)`: distancia menor que `d_coast`; PI com referencia zero e
  saturacao negativa, aplicado como comando reverso de frenagem no ESC.
- `HOLD (3)`: parado ou com inversao de movimento; neutro. A retomada exige
  distancia de parada da velocidade desejada, margem adicional e 0.5 s
  consecutivos com caminho livre.
- `FAULT (4)`: encoder invalido, ou ultrassom invalido com o carro parado;
  neutro e bip, sem retomar enquanto os sensores nao forem validos.

Uma intervencao permanece ativa ate parar. Durante coast-down, uma reducao
da distancia disponivel pode promover `COAST` para `BRAKE`. Depois de
entrar em `BRAKE`, o ADAS mantem a frenagem ate parar. Ultrassom invalido
com velocidade valida positiva solicita frenagem preventiva. Encoder
invalido corta a tracao: sem velocidade confiavel, nao e possivel garantir
a retirada do reverso no instante certo.

O freio reutiliza `KP_VEL`, `KI_VEL` e o anti-windup do PI existente. A
saida `u` e negativa durante a frenagem, sem referencia negativa de
velocidade. `Servos.set_brake()` aplica esse PWM de imediato, sem usar o
procedimento normal de marcha re, que espera 5 s e depois envia pulsos.
O estado `gear` continua representando a marcha de conducao. Quando o
encoder bruto indica velocidade <= `stop_speed`, o comando volta a neutro.

## Configuracao e calibracao

Os parametros ficam no dicionario `parameters['adas']` do `main.py`; os
valores restantes sao definidos em `BrakingConfig`.

| Parametro | Inicial | Unidade |
| --- | ---: | --- |
| `coast_deceleration` | `0.5 * CAR['MI'] * CAR['GRAV']` = 0.1962 | m/s^2 |
| `clearance` | 0.20 | m |
| `reaction_time` | 0.40 | s |
| `anticipation_time` | 0.30 | s |
| `stop_speed` | 0.05 | m/s |
| `release_margin` | 0.15 | m |
| `release_time` | 0.50 | s |
| `max_brake` | 1.0 | fracao do throttle reverso calibrado |

`coast_deceleration` e uma estimativa inicial do modelo, nao uma calibracao
dos carrinhos. Obtenha a desaceleracao com motor neutro em diferentes
velocidades e use um limite inferior observado, incluindo variacao de piso
e carga. A 1 m/s, o valor inicial estima `d_coast` em aproximadamente
3.15 m, incluindo a margem de 0.20 m e o atraso de 0.40 s.

`reaction_time` inclui o prazo de validade de 0.30 s dos sensores, atraso
do filtro de velocidade, ciclo de controle e atualizacao do PWM. Aumente-o
se os atrasos medidos forem maiores. A margem e medida a partir do sensor;
inclua a distancia do sensor ate a frente do carro na escolha de `clearance`.

Antes de ensaio no chao, confirme em bancada o sinal do encoder e que o
ESC responde ao primeiro comando abaixo do neutro com frenagem/reverso
contrario ao movimento. O modelo/configuracao do ESC nao esta registrado
no repositorio; o codigo nao presume que os pulsos de armar a marcha re
sejam necessarios para frear. Calibre tambem `max_brake` e verifique a
distancia real de frenagem: o ADAS nao garante parada em uma distancia
fisicamente insuficiente, nem quando o sensor nao detecta o obstaculo.

O mecanismo e frontal e recebe referencias >= 0; nao protege marcha re.

## Registros

O terminal/GUI exibe linhas `ADAS,...` nas mudancas de estado ou validade
dos sensores. O CSV mantem `u` com sinal e acrescenta `v_raw`,
`velocity_valid`, `adas_mode`, `obstacle_distance`, `obstacle_valid`,
`coast_stop_distance` e `adas_required_deceleration`. Decisao e comando
sao atualizados na mesma amostra de sensores, sem duplicar o tempo.

## Testes sem hardware

```sh
python3 -m unittest discover -s veiculo_real/tests -v
```

Os testes precisam de NumPy. Os drivers de I2C/serial sao substituidos
apenas na importacao; a decisao ADAS, o PI e a atuacao PWM sao reais.
