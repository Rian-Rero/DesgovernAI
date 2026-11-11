# Plano de correção: centralização, oscilação, ultrassom e linha amarela

Base: 11 rodadas de 10/11 (`Archive.zip`, ~1.300 quadros originais gravados) +
pista virtual. Cada número abaixo vem desses logs.

## 1. Diagnóstico

### 1.0 O carro rodou código antigo
Nenhuma das 11 rodadas tinha as correções dos últimos dias:

| Correção | Estava no carro? |
|---|---|
| Lado das cores pelo corredor (troca errada em grampo) | não (2 rodadas trocaram de lado em movimento) |
| Confirmação do eco do ultrassom (0,15 s) | não |
| Trim da direção aprendido | não |
| Curva mais fechada R = 1,2 m | não |
| Pure pursuit | só 1 de 11 (as outras com Stanley) |

Hoje não há como saber qual versão rodou: o log não grava isso.

### 1.1 Ultrassom (9 de 11 rodadas pararam)
Todas as 9 paradas foram **uma única leitura**: de 4,0 m (nada) para 0,4–0,66 m, e
de volta a 4,0 m na leitura seguinte. Com o carro já parado, o sensor seguia
oscilando. A confirmação de 0,15 s (2 leituras), que já está no código, elimina 7
das 9. Nas outras 2, o carro estava de fato colado nos cones (problema 1.2).

### 1.2 Carro fora do centro e batendo
A pista desses ensaios tem uma **linha amarela no centro**: uma régua de verdade.
Nos 217 quadros em que a linha detectada está entre os cones:

- **O carro anda ~10 cm à direita da linha** (mediana), e em 10% do tempo a mais de
  31 cm dela. A pista tem 0,97–1,2 m, então a folga de cada lado é de 35–47 cm.
- **O controle não leva o carro ao centro que ele mesmo estima**: o centro
  estimado fica ~20 cm à esquerda do carro, com a direção 3–7° para a esquerda em
  média nas retas. Causas: Stanley (10/11 rodadas), viés da direção (sem trim) e
  atraso da visão (1.3).
- **O próprio centro estimado erra ~10 cm**: pela câmera, a linha fica a 0,65 m dos
  cones da esquerda e a 0,53 m dos da direita. É o erro da **calibração**, que só
  tem pontos até 1,65 m e ±0,3 m, enquanto os cones ficam a ±0,55–0,65 m.

### 1.3 Visão lenta (raiz da oscilação em velocidade)

| | Medido |
|---|---|
| Quadros por segundo | 5,6–9,3 |
| Idade da imagem no controle | 260–290 ms (p90 até 420 ms) |
| Captura / detecção / geometria | 38 / 95 / 17 ms |
| Gravando quadros (`--record-frames`) | −20% de quadros/s |

A 1 m/s, o carro anda ~30 cm entre a foto e o uso dela. A compensação de atraso
só cobre até 0,25 s, e a foto inteira (900 kB) atravessa os processos a cada
quadro.

### 1.4 Oscilação acima de 0,7 m/s
Com o carro devagar, o rumo varia 4–5° RMS; a 0,7–1,0 m/s, 12–19° RMS, e o erro
lateral sobe de 4–8 para 12–15 cm. Na pista virtual, com o atraso real (7 q/s,
0,3 s), o Stanley erra o dobro do pure pursuit (8,8 contra 4,0 cm RMS; máximo de
24 contra 11 cm).

### 1.5 Detecção
- Falsos cones em mochilas e roupas vermelhas ou laranja.
- Borrão de movimento em velocidade (pouca luz: exposição longa).
- Corredor estreito, com objetos encostados nas paredes.

### 1.6 Linha amarela
A fita é **fina e tracejada**, e o piso tem **reflexo da luz quente no mesmo
matiz**:

| | Saturação (mediana) | b* do Lab (mediana) |
|---|---|---|
| Fita | 49 | 14 |
| Reflexo do piso | 30 | 8 |

O limiar atual (S ≥ 25) aceita os dois, e o detector segue o reflexo. Quando a
fita é contínua e perto, a vista de cima a acha certo.

## 2. Plano

### Etapa 0. Garantir que o código certo roda (código, 30 min)
- Carimbo de versão: o `main.py` calcula um hash dos `.py` e o mostra no início.
  Ele também é gravado no `cones_summary.json` e no `car.csv`.
- A GUI mostra a versão local e a do carro antes de executar, e avisa se forem
  diferentes.
- Critério: todo log diz qual versão rodou.

### Etapa 1. Visão rápida (código; meta ≥ 15 q/s e idade ≤ 150 ms)
1. Processar a 320×240 (`vision_width 320`). No Pi, a detecção cai de ~95 para
   ~30 ms. Os cones até 3 m são os mesmos que em 640 (benchmark de 10/11).
2. Gravação de quadros só em coleta, nunca em corrida: na corrida, no máximo
   1 Hz, num processo de baixa prioridade.
3. O processo de visão manda ao controle só o resultado. A foto vai só quando a
   transmissão ou o debug pedirem, reduzida.
4. Câmera configurada já em 320×240, se a webcam entregar: menos USB e menos
   decodificação.
- Critério: `vision_age_s` ≤ 150 ms (mediana) e ≥ 15 quadros novos/s no `car.csv`.

### Etapa 2. Controle que centraliza (código)
1. Pure pursuit como padrão em todo lugar (já é na GUI); Stanley só como
   reserva.
2. Compensação de atraso exata: o estado do estimador é levado até o instante do
   comando pela odometria, com a clotoide inteira, até 0,5 s (hoje: curvatura
   constante, até 0,25 s).
3. Trim da direção aprendido (já no código), com um procedimento de 1 minuto:
   3 passadas numa reta, para o trim estabilizar antes do treino.
4. Velocidade limitada pelo atraso: v ≤ 0,3 m / idade da imagem. A 0,3 s, isso
   dá ≤ 1 m/s; com a etapa 1, sobe sozinho.
5. Alvo do pure pursuit ≥ v × idade + 0,3 m, para nunca mirar num ponto que a
   imagem já "passou".
- Critério (pista virtual com atraso real e no carro): erro lateral ≤ 5 cm RMS,
  rumo ≤ 6° RMS a 1 m/s.

### Etapa 3. Calibração da câmera (pista, 45 min; eu preparo a ferramenta)
1. Grade de fita no chão: x = 0,6 / 1 / 1,5 / 2 / 2,5 / 3 m e y = −1 / −0,5 /
   0 / +0,5 / +1 m (30 pontos).
2. Foto pela GUI, no notebook: clicar nos pontos gera o `camera_lateral.json`,
   com o erro de cada ponto em cm. A ferramenta atual abre janela no Pi, o que
   não funciona pelo hotspot.
3. Validação automática com a linha amarela: os cones devem ficar simétricos em
   relação à linha.
- Critério: erro de reprojeção ≤ 3 cm até 2,5 m; cones a ±2 cm da simetria em
  relação à linha.

### Etapa 4. Ultrassom (código pronto; falta validar)
- Confirmação de 2 leituras (0,15 s) e filtro de curva já estão no código.
- Acrescentar: leitura 0,00 m é falha do sensor, não obstáculo.
- Critério: 0 paradas sem obstáculo real numa sessão de 10 rodadas, e parada
  real comprovada com uma caixa no meio da reta.

### Etapa 5. Detecção de cones robusta (código + 2–3 h de rotulagem)
1. Rede classificadora treinada com os ~1.300 quadros reais já gravados.
   Rotular 200–300 tira os falsos em mochila e roupa.
2. Faixa de busca: só entram cones a W/2 ± 0,3 m do corredor previsto. Hoje o
   descarte só olha a distância do cone ao centro, sem limitar a faixa.
3. Exposição: limitar a exposição automática (sem queda de quadros/s), com a
   correção de luz do detector compensando o brilho; menos borrão.
- Critério: falsos cones/quadro ≤ 0,1 nos quadros rotulados que ficaram fora do
  treino.

### Etapa 6. Linha amarela (código)
1. Cor relativa ao piso: a vista de cima normalizada pela cor do piso (como nos
   cones) e exigência de S ≥ 40 e b* ≥ b* do piso + 5. O reflexo quente sai.
2. Tracejado: unir traços na vista de cima (gap de até 0,4 m).
3. Linha e cones juntos (a pista tem os dois): o estimador usa a linha como
   centro e os cones como bordas, e assim centraliza melhor do que cada um
   sozinho.
4. Validação nos ~1.000 quadros reais com linha.
- Critério: ≥ 90% dos quadros com a linha detectada entre os cones e a ±5 cm do
  meio corrigido.

### Etapa 7. Medir cada ensaio do mesmo jeito (código)
`tools/analisar_ensaio.py logs/<ensaio>` gera o relatório que montei à mão hoje:
- versão do código;
- quadros por segundo e idade da imagem;
- paradas (falsas ou reais);
- erro em relação à linha amarela;
- oscilação do rumo;
- trim.

## 3. Ordem

| Quando | Quem | O quê |
|---|---|---|
| Agora | Eu | Etapas 0, 1, 2, 4, 6 e 7 (código), validadas na pista virtual e nos quadros reais de 10/11 |
| Próxima ida à pista (~2 h) | Vocês | Enviar → conferir a versão → calibração (etapa 3) → 3 passadas de trim → ensaio de 10 rodadas a 0,5 / 0,7 / 1,0 m/s → Coletar Dados |
| Depois | Vocês (rotular) + eu (treinar) | Etapa 5 (rede) |

Para fechar, o critério do ensaio: **10 rodadas seguidas sem bater e sem parada
falsa, a 0,7 m/s**, com o carro a ≤ 5 cm RMS da linha amarela. Depois, subir
para 1,0 m/s.

## 4. Situação em 11/11 (etapas de código feitas)

Testes: 231 do veículo OK. Em `tools/tests`, só a falha antiga
`test_repository_calibration_is_flagged_as_short` (a calibração do repositório
não cobre os cones).

| Etapa | Feito | Medido |
|---|---|---|
| 0. Versão | `fva_car/version.py`; o `main.py` imprime e grava a versão; a GUI compara notebook × carro antes de rodar | — |
| 1. Visão rápida | `vision_width` 320 (main e GUI). A observação já ia sem as máscaras; a foto restante custa ~2 ms no pipe e ficou | 617 quadros reais: 98/94/86% dos cones de 640 px (até 1 / 1–2 / 2–3 m), 2,5× mais rápido |
| 2. Controle | Pure pursuit padrão; compensação exata (clotoide, até 0,5 s); v ≤ 0,3 m ÷ idade da imagem; **trim pelo esterço médio** (novo, abaixo) | Pista virtual com 2° de viés, 1 corrida, trim do zero: antes 6,6–13,2 cm RMS (máx. 35); agora 5,8–8,8 (máx. 26) com a visão lenta e 4,4–9,9 (máx. 21) a 320 px |
| 3. Calibração | Botão **📷 Foto p/ calibração** na GUI e grade clicável no notebook, com o erro de cada ponto deixado de fora do ajuste (`tools/calibrar_camera_lateral.py`) | Calibração atual: 0,4–3,8 cm por ponto, só na área medida |
| 4. Ultrassom | Leitura < 3 cm é falha, não obstáculo | Nos logs: as 9 paradas eram leituras isoladas |
| 6. Linha amarela | Cones apagados, borda da vista cortada, ±1,0 m, H ≤ 35, saturação ≥ 36; fita + cones juntos no estimador, um quadro sim e um não | Fita certa em 86% dos quadros em que é achada (antes 46%); fusão: mesmos quadros válidos, +6 ms por quadro em média |
| 7. Relatório | `tools/analisar_ensaio.py <pasta> [--linha]` e botão **📋 Relatório** na GUI; aviso de que gravar quadros deixa a visão mais lenta | — |

| 5.2 Faixa de busca | Cone a mais de 0,2 m (+ 0,1/m além de 1 m) da borda **ajustada** sai (`band_m`, `band_growth`) | Pista virtual com 15 objetos fora da pista: 11/12 → 12/12 voltas, 11,1 → 4,4 cm RMS, 2 → 0 toques. Real: 22 de ~11 mil cones tirados |

Não feito: gravar quadros num processo de baixa prioridade (1.2) e abrir a
câmera já em 320×240 (1.4). A detecção a 320 px já resolve o grosso, e a câmera
a 320 mudaria a calibração (pixels de 640×480).

### Revisões do diagnóstico

- **Calibração (1.2)**: medindo pela fita amarela (`analisar_ensaio --linha`),
  o centro dado pelos cones fica a **1–4 cm da fita a 1 m** nas 3 rodadas com
  fita. O erro de ~10 cm era menor do que parecia. Refazer a calibração com a
  grade continua valendo (cobre os cones e até 3 m), mas não é a causa
  principal.
- **Viés da direção (novo)**: sem IMU, a guinada da odometria vem do esterço
  comandado. Com viés na direção, a odometria entorta e o estimador também. Na
  pista virtual, com 2°, o carro ficava 11 cm fora do centro com o estimador
  vendo −2 cm. Por isso o trim aprendido pelo erro lateral não funcionava (e
  em pista curta aprendia no sentido errado). Agora o trim vem do **esterço
  médio na reta** (`centering_steer_gain` 0,3; `centering_gain` 0). Com o trim
  já certo, os 2° somem (4,0 cm RMS, igual a sem viés). O trim é salvo entre
  corridas: **as primeiras rodadas de cada dia servem para ele assentar**.

### Na pista (próxima ida)

1. Enviar e conferir "✅ Código igual" na GUI.
2. Grade de fita → **📷 Foto p/ calibração** → clicar → salvar → Enviar.
3. 3 passadas numa reta a 0,5 m/s (o trim assenta; o terminal mostra o valor).
4. 10 rodadas a 0,5 / 0,7 / 1,0 m/s, sem gravar quadros, e uma com quadros
   para medir pela fita.
5. Coletar Dados → **📋 Relatório** em cada uma. As metas aparecem como
   OK/ATENCAO.
