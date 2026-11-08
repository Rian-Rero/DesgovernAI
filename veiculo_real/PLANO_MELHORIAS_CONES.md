# Plano de melhorias: cones, trajetória, seguimento e velocidade (Raspberry Pi 4)

Base: ensaios `logs/20261023_080354` (E1) e `logs/20261023_080437` (E2), código
da branch `latero-lara-c` (commit `b5ca8d8`) e perfil do detector em quadros reais.

---

## 1. Diagnóstico (o que os logs mostram)

| # | Problema | Evidência | Causa no código |
|---|----------|-----------|-----------------|
| D1 | **O carro não sai do lugar** (E1) | 182 de 186 quadros inválidos com "Cones sem cobertura métrica junto ao eixo dianteiro". O centro estava ótimo (y ≈ 0,02 m, reta), mas começava em x = 1,86 m. | `fit_line_errors` recusa o centro se `min(x) > 1,5 m` (`fva_car/lateral.py:316`). Com o FOV da câmera, o primeiro par de cones só aparece a cerca de 1,3–1,9 m: a ±0,9 m de lado, um cone só entra na imagem para x ≳ 1,6 m. **O Stanley exige erro no eixo dianteiro, uma região que a câmera não vê.** |
| D2 | **Extrapolação até o eixo gera erros absurdos** | E2, quadro 101: centro à **esquerda** (+0,40 m a 0,9–1,4 m), Stanley recebe `cross_track = −0,23` e `heading = +0,87 rad`. O carro fechou a curva e quase bateu no cone interno (0,72 m, −0,22 m). Isso disparou a parada por obstáculo. | O ajuste por parábola ou arco é extrapolado de x ≥ 0,9 m até x = 0 (`lateral.py:340-376`). |
| D3 | **Visão lenta** | 6,5 fps (E1) e 8,6 fps (E2), com detecção de 115–153 ms. A idade da imagem no comando fica entre 0,25 e 0,30 s. | Perfil: `_bases` consome **68%** do tempo, com ~136 chamadas de `contact()` e ~160 `connectedComponents` por quadro, porque o *split* por 8 erosões roda em manchas de ruído. `segment` consome ~12% (Lab float + `cartToPolar` + máscara fraca dupla + `medianBlur`). |
| D4 | **Cones próximos não detectados** | Na `vision_detection.jpg`, os 3 cones verdes grandes à esquerda e o vermelho mais próximo à direita não foram detectados. Os detectados estão quase todos longe (linhas 270–295, 2–3 m). Rejeições em E2: vermelho `pequeno` 1344, `cortado_borda` 291, `tamanho` 256. | O cone cortado na lateral é descartado mesmo com a base visível. O `size_check` usa a homografia extrapolada (calibração só até 1,65 m e ±0,30 m). Os cones na sombra perdem saturação. |
| D5 | **Falsos positivos de cor** | Muitos cones `px` sem projeção `m`. A borda verde do E2 dá a volta pela frente do carro (y de +1,2 a −1,3 m) e uma "borda esquerda" vai parar na direita. Em seguida, o centro virtual fica a 1 m de distância. | Não há restrição geométrica de que a borda fique do seu lado nem de quanto ela pode girar. Uma cor isolada pode iniciar uma cadeia. |
| D6 | **Calibração / checagem incoerente** | O aviso "Calibração incompatível" aparece em ~50 quadros: espaçamento de 0,48 m (esperado 0,70) e largura de 1,6–1,9 m (esperada 0,90). A largura medida é **consistente** (1,7–1,9 m nos dois ensaios), e a foto mostra um corredor de ~2 m. | `track_width_cm=90` e `cone_spacing_cm=70` (`tools/cone_path.py:72-73`) parecem ser valores do simulador. Com o aviso ativo, `calibration_trusted()` desliga o `size_check`, o que **piora os falsos positivos**. A homografia tem 12 pontos só entre 0,75–1,65 m e ±0,30 m, mas é usada até 3 m e ±1,3 m. |
| D7 | **vref oscilante** | E2: vref de 0,19 a 0,47 m/s, com 36 inversões de dv. O *hold* alterna 28 vezes em 14 s (36% do tempo em hold). O fator de hold faz dente de serra entre 0,55 e 1,0. | (a) `curve_speed` divide por `1 + 2·|e_y| + |e_ψ|` (`lateral.py:380-388`), e esses erros são justamente os extrapolados e ruidosos de D2. (b) A rampa de hold reinicia a cada quadro válido. (c) O limite de curva usa a curvatura de um único quadro (de −0,86 a +0,25 entre quadros). (d) A rejeição de salto de 0,12 m é menor que o ruído métrico a 2–3 m: houve 31 eventos "salto" e cada um aciona o hold. |
| D8 | **Velocidade real oscila mesmo com vref constante** | 2–3 s: vref de 0,25–0,28, mas v vai de 0,00 a 0,23 m/s. | Zona morta ou atrito estático do motor em baixa velocidade. O FF é linear (`u = 0,17·v`) e não tem offset de partida (`fva_car/car.py:42-44`). |
| D9 | **Direção satura** | 18% do tempo em ±20°. | Consequência de D2 e D6 (erro lateral superestimado) e do ganho Stanley de 1,5 sobre erros extrapolados. |
| D10 | **Odometria só cinemática** | `w_imu = nan` (IMU comentada no commit `1ae7dc0`). | O yaw vem de `v·tan(δ)/L` com o servo. O mapa de cones e a compensação de latência dependem desse yaw. |

---

## 2. Arquitetura proposta (visão geral)

```
 [Processo VISÃO]                                  [Processo CONTROLE 50 Hz]
 câmera (exposição/WB fixos, MJPG 640x480@30)       odometria (encoder + giroscópio)
   -> ROI do piso (linhas horizonte..480)            -> pose integrada
   -> LUT de cor (1 passada, sem float)               |
   -> componentes + filtro vetorizado + forma         |
   -> bases (px) -> piso (m) com covariância  ===>  MAPA LOCAL DE CONES (odom, estático)
      (memória compartilhada, só o mais recente)      -> bordas E/D (restrições geométricas)
                                                      -> centro (portões / 1 borda + W)
                                                      -> PURE PURSUIT no ponto à frente
                                                      -> perfil de velocidade (preview + slew)
```

Princípios:
1. **Cones são marcos estáticos.** Ficam num mapa local no referencial da odometria e têm a posição média de vários quadros. O ruído cai com √N e o cone continua conhecido depois de sair do FOV. Isso resolve grande parte do problema de "só vejo uma cor".
2. **Controlar no que a câmera vê.** Usar *pure pursuit* com um ponto à frente (lookahead de 0,9–1,6 m), sem extrapolar até o eixo.
3. **Um único estimador suaviza.** O mapa e o ajuste do centro substituem as três camadas atuais de remendo: rejeição de salto, hold e mistura de caminhos.
4. **O tempo da visão nunca bloqueia o controle.** Usar um processo separado, já que o GIL hoje é compartilhado por captura, visão, gravação e controle.

---

## 3. Fases

### Fase 0: Medição base (antes de mexer) · ~1 dia
- [ ] **Medir a pista real**: largura entre fileiras, espaçamento entre cones e altura e base do cone. Corrigir `track_width_cm`, `cone_spacing_cm`, `cone_height_cm` e `cone_base_cm`.
- [ ] Gravar um ensaio de coleta com `--record-frames 3`, com o carro empurrado à mão pela pista inteira, incluindo as curvas e as sombras.
- [ ] Rotular ~150 quadros com `tools/rotular_cones.py`. Medir precisão e revocação **por faixa de distância** (<1,5 m, 1,5–2,5 m, >2,5 m) com `tools/avaliar_cones.py`. Esse conjunto passa a ser o teste de regressão de toda mudança de detecção.
- [ ] Medir o tempo no Pi 4 com `tools/benchmark_visao.py`: captura, segmentação, bases, geometria e o fps nativo da câmera com exposição automática e com exposição fixa.

**Critério**: tabela base de P/R por distância, ms por etapa e fps da câmera.

### Fase 1: Calibração e câmera · ~0,5 dia
- [ ] **Recalibrar a homografia** com 20 ou mais pontos no piso cobrindo **0,6–3,0 m em x e ±1,3 m em y**, a área onde os cones são usados. Hoje são 12 pontos em 0,75–1,65 m e ±0,30 m. Reportar o erro de reprojeção por ponto e rejeitar a calibração se o erro passar de 3 cm a 2 m.
- [ ] Avaliar **inclinar a câmera alguns graus para baixo**. Hoje o horizonte fica na linha ~256 e metade da imagem é teto e parede. Com mais piso na imagem, os cones próximos entram com mais pixels. Recalibrar depois.
- [ ] **Fixar a exposição e o balanço de branco** (`CAP_PROP_AUTO_EXPOSURE`, `CAP_PROP_AUTO_WB`, `CAP_PROP_EXPOSURE`) e pedir MJPG (`CAP_PROP_FOURCC`). A exposição automática em ambiente interno escuro derruba o fps da câmera e muda o matiz entre quadros, o que exige limiares HSV largos e gera falsos positivos.
- [ ] Trocar a "checagem de calibração" por valores medidos, ou desligá-la. **Ela nunca deve desligar o `size_check`.**

**Critério**: um cone colocado em (2,0 m; ±0,9 m) deve ser medido com erro abaixo de 5 cm.

### Fase 2: Desempenho do pipeline de visão · ~2 dias
Meta no Pi 4: **detecção ≤ 30 ms, visão ≥ 20 Hz e idade da imagem no comando ≤ 0,10 s.**

1. **Segmentação por LUT** (`tools/cone_path.py:220-267`):
   - Pré-calcular na inicialização uma LUT `uint8` de 32×32×32 (BGR >> 3) com as classes {0, verde forte, verde fraco, vermelho forte, vermelho fraco}, já combinando as regras HSV, o ângulo e o croma Lab e a histerese.
   - Por quadro, só `lut[b>>3, g>>3, r>>3]` sobre a ROI. Isso substitui 2× `cvtColor`, `cartToPolar` float, 4× `inRange`, a máscara fraca dupla e os `bitwise`.
   - Trocar `medianBlur` no BGR por `MORPH_OPEN` 3×3 na máscara binária, que é mais barato e remove os pontos isolados que geram as ~1000 rejeições "pequeno".
2. **Filtro vetorizado antes do Python** (`_bases`, `cone_path.py:270-422`). Com `stats` de `connectedComponentsWithStats`, descartar de uma vez, via numpy:
   - área fora de [0,3; 3]× a área esperada na linha da base, usando uma tabela por linha pré-calculada de `row_scale`;
   - proporção h/w fora de [0,8; 4];
   - base acima do horizonte;
   - largura maior que 2× a largura esperada.

   Só os sobreviventes (tipicamente menos de 15) vão para `contact()`.
3. **Remover o split por 8 erosões.** Para uma mancha larga demais (dois cones unidos), dividir pelo mínimo do perfil de colunas (`mask.sum(axis=0)`): uma operação, sem recursão.
4. **Cadeias no piso, não em pixels.** O `_chains` O(n²) em Python com até 80 pontos passa a operar sobre os cones do mapa (Fase 4), que são poucos, já em metros.
5. **Processo separado para captura e visão** (`multiprocessing` + `shared_memory`, só com o resultado mais recente). Usar `cv2.setNumThreads(2)`. O controle e o gravador de logs ficam no processo principal. A gravação de JPG de eventos e o *stream* continuam fora do caminho crítico.
6. Testes: os de `tests/test_vision_pipeline.py` e `tools/tests/test_cone_candidates.py` precisam continuar passando, e a P/R da Fase 0 não pode cair.

**Critério**: o benchmark no Pi mostra ≤ 30 ms por quadro e o log de pista mostra `vision_fps ≥ 20`.

### Fase 3: Detecção robusta (menos falsos positivos, mais cones próximos) · ~2 dias
- [ ] **Aceitar cone cortado na lateral se a base estiver visível.** Hoje `cortado_borda` rejeita se `x+foot.min() <= 0` (`cone_path.py:332`). Passar a aceitar quando a linha de base está dentro da imagem e o corte é só lateral, marcado como `parcial`, com peso menor no mapa.
- [ ] **Validação geométrica no piso** para todo candidato: 0,3 ≤ x ≤ 3,0 m, |y| ≤ 1,5 m, e altura em pixels compatível com a altura do cone na linha da base (±40%). Manchas em parede, porta, roupa e guarda-chuva falham nesse teste porque a "base" não fica no plano do chão com a altura certa.
- [ ] **Confirmação temporal com o mapa.** Um candidato novo só vira cone depois de 2 observações a menos de 15 cm, no referencial da odometria. O `confirm_hits=2` já existe e passa a valer no mapa persistente. Um cone que deveria estar no FOV e não é visto por 3 quadros perde confiança.
- [ ] **Cor robusta à sombra.** A LUT com classe "fraca" só vale se a componente tocar a classe forte (histerese, como hoje). Verificar na P/R se os cones verdes escuros próximos passam.
- [ ] (Opcional, se a P/R continuar ruim) Usar um classificador leve no recorte do candidato, por exemplo HOG + SVM linear ou uma CNN minúscula em 24×32. O `outros/best.pt` (YOLO) é pesado demais para 20 Hz no Pi 4, mas pode servir para **gerar rótulos** offline.

**Critério**: revocação ≥ 90% para cones a menos de 2,5 m e precisão ≥ 95% no conjunto rotulado.

### Fase 4: Mapa local, bordas e trajetória · ~3 dias
**4.1 Mapa de cones** (evolução de `tools/cone_tracking.py`)
- Cada cone tem estado `(x, y)` no referencial **odom**, covariância, cor, número de observações e instante da última observação.
- **Predição**: com a pose da odometria. **Atualização**: média ponderada pela covariância da medida, com σ crescendo com a distância (σ ≈ 0,02 + 0,015·x² m) e maior para cones `parcial`.
- **Retenção por distância, não por tempo**: o cone fica até estar 0,5 m atrás do eixo traseiro (x_carro < −0,9 m). Hoje o limite é 350 ms ou 2,5 s, o que falha quando o fps cai.
- **Habilitar o giroscópio da IMU** para o yaw (`fva_car/car.py:381`, comentado). Com um yaw ruim, o mapa "entorta" nas curvas.

**4.2 Bordas com restrições geométricas** (corrige D5)
- A borda esquerda começa pelo cone mais próximo com y > 0 e a direita com y < 0, no referencial do carro.
- Ao encadear: passo de 0,25–1,2 m e giro de no máximo 45° entre segmentos e 120° no total. A borda **não pode cruzar a outra** nem passar pela frente do carro (|y| < W/2 com x < 1 m).
- Cor ↔ lado: decidir com as duas cores visíveis, com histerese, como hoje. Com uma cor só, **manter o último mapeamento**.

**4.3 Centro**
- **Duas bordas**: portões (pares normais) e ajuste de spline suavizada com peso por covariância, como hoje com `gate_midpoints` + `fit_path`, mas sobre o mapa.
- **Largura W**: média móvel das larguras medidas com as duas bordas, limitada a [0,8; 1,2]× a largura nominal medida na Fase 0. Persiste a corrida inteira, não 1,5 s.
- O centro é mantido no referencial odom e vale também **atrás** da parte visível, porque o mapa lembra os cones já passados. Isso elimina D1: o centro sempre cobre o eixo.

**4.4 Critérios com uma cor só** (o pedido principal)

Estado da percepção, avaliado a cada quadro com histerese de 2 quadros:

| Estado | Condição | Como estimar o centro | Velocidade |
|--------|----------|-----------------------|------------|
| `AMBAS` | ≥ 2 cones de cada lado no trecho 0–2,5 m (mapa) | Portões | Teto cheio |
| `SO_EXTERNA` | Uma borda com ≥ 3 cones, curvando **para o lado oposto** ao seu (borda direita com κ > 0 = curva à esquerda) | Desloca a borda W/2 pela normal para dentro. Curvatura do centro: **κc = κb / (1 − |κb|·W/2)**, de módulo maior que a da borda, para virar mais. Viés extra de **0,05–0,10 m para dentro** (b = 0,1·W·min(1, \|κb\|/κ_ref)). | ×0,8 |
| `SO_INTERNA` | Uma borda com ≥ 3 cones, curvando **para o seu lado** | Desloca W/2 para fora: κc = κb / (1 + \|κb\|·W/2). Folga mínima de W/2 − 0,10 m de **cada cone interno** (restrição dura). | ×0,8 |
| `UMA_BORDA_RETA` | Uma borda com \|κb\| < 0,15 1/m | Paralela à borda a W/2 | ×0,85 |
| `MEMORIA` | Menos de 3 cones visíveis, mas o mapa ainda tem a trajetória à frente (≥ 0,6 m) | Segue o centro do mapa. **Em curva, mantém a curvatura atual em vez de endireitar**: o centro é prolongado como arco de κ = último κc filtrado, até 0,8 m. | ×0,6 |
| `PERDIDO` | Distância percorrida em `MEMORIA` > 0,8 m, ou nenhum dado | Neutro e parada (como hoje) | 0 |

Regras complementares:
- **Ao entrar numa curva à esquerda e perder os cones internos (verdes)**, o estado passa de `AMBAS` para `SO_EXTERNA`. Como κc > κb, o *feedforward* e o pure pursuit viram mais que a borda vermelha indica. O mapa ainda tem os verdes já vistos ao lado do carro, então W e o lado continuam conhecidos.
- **Decidir externa ou interna exige a curvatura da borda com ≥ 3 cones.** Com 2 cones, usar o sinal de κ do estado anterior. Se não houver estado anterior, tratar como `UMA_BORDA_RETA`.
- **Cone isolado novo**, de cor oposta, dentro do corredor previsto, a menos de 1,2 m: é um cone interno de curva aparecendo. Ele atualiza a borda, não é ruído.
- **Tolerância por distância, não por quadros**: os hoje 6 quadros a 8 fps (0,75 s) seriam outra coisa a 20 fps. Usar metros percorridos (0,8 m) e um teto de tempo (1,0 s).

**Critério**: em replay dos logs E1 e E2, nenhum quadro com borda cruzando a frente, E1 deve partir, e o centro deve ser contínuo (salto < 5 cm entre quadros).

### Fase 5: Seguimento e centralização · ~1,5 dia
- [ ] **Pure pursuit** sobre o centro do mapa (substitui o Stanley no modo cones e o mantém para a linha):
  - L_d = clamp(0,9 + 0,5·v, 0,9, 1,6) m, medido sobre o arco do centro a partir do ponto mais próximo do eixo traseiro;
  - δ = atan(2·L·sin α / L_d) + k_ff·atan(L·κc(s_preview)), com k_ff entre 0,3 e 0,6, porque o PP já incorpora boa parte da curvatura;
  - limites de ±20° e taxa de 120°/s mantidos (`main.py:51-55`).
- [ ] **Centralização**: um termo integral lento no erro lateral **medido no ponto visível mais próximo**, não no eixo (Ki pequeno, |I| ≤ 2°, zerado em curva com |κ| > 0,5). Ele corrige o desvio de montagem da câmera: o y = 0 da calibração cai na coluna ~240, não em 320.
- [ ] **Restrição de folga**: se algum cone do mapa estiver a menos de 0,15 m do corredor varrido pelo carro nos próximos 0,8 m, deslocar o ponto alvo para o lado oposto. É uma proteção contra o caso D2.
- [ ] A compensação de latência continua (`MAX_LATENCY_COMPENSATION`), agora aplicada à pose sobre o mapa, o que é trivial porque o mapa é odom.
- [ ] Remover a rejeição de saltos `cone_max_lateral_jump_m` / `heading_jump`, que viram redundantes com o mapa, e o *search steering*, substituído pela regra de manter a curvatura em `MEMORIA`.

**Critério**: em reta, |e_y| RMS < 5 cm. Em curva, nenhum cone tocado. Saturação da direção < 5% do tempo.

### Fase 6: Referência de velocidade · ~1 dia
Novo `speed_profile(state)` calculado **só a cada novo resultado de visão** e suavizado no laço de 50 Hz:

```
kappa_prev = max |κc(s)| para s em [0, v·0,6 + 0,6] m à frente        # preview, não 1 quadro
v_curva    = sqrt(a_lat_max / max(kappa_prev, 1e-3))                     # a_lat_max 0,5 m/s²
v_conf     = v_set · {AMBAS 1,0; UMA_BORDA 0,8; MEMORIA 0,6; PERDIDO 0}
v_erro     = v_set / (1 + 1,5·max(0, |e_y_vis| − 0,10))                  # zona morta, sem heading
v_alvo     = clamp(min(v_set, v_curva, v_conf, v_erro), v_min, v_set)
# histerese: só muda v_alvo se |Δ| > 0,03 m/s
# limitador de taxa assimétrico no laço 50 Hz: +0,3 m/s² (acelera), −1,0 m/s² (reduz)
```

- Remove o termo `|heading|` do `curve_speed` (`lateral.py:386`), que é a maior fonte de oscilação, e a rampa de hold que reinicia a cada quadro.
- `v_min` operacional ≥ 0,30 m/s, acima da zona morta (D8).
- **PI de baixo nível** (`fva_car/car.py:438-471`): acrescentar um offset de partida `u0` ao FF, `u = u0·[v_ref > 0] + 0,17·v_ref + PI`, identificado com os ensaios `simulador/ensaios/dados/banguela_*`. Também reavaliar o filtro de velocidade do encoder, por período entre pulsos e não por média móvel.

**Critério**: em reta, o desvio-padrão de vref é menor que 0,02 m/s. Em curva, vref cai antes de entrar (preview) e só sobe na saída. O tempo em hold cai abaixo de 5%.

### Fase 7: Validação e ferramentas · contínuo
- [ ] **Replay offline**: rodar o pipeline novo sobre `cones_debug.jsonl` + frames gravados, comparando estados, centro e vref com o log original. Isso permite iterar sem o carro.
- [ ] Testes unitários novos: estados da tabela 4.4 (curva esquerda com só a externa → κc > κb), borda que não cruza a frente, LUT igual às regras HSV/Lab, e perfil de velocidade monotônico com o preview.
- [ ] Indicadores no `cones_summary.json`: fps, p95 da detecção, % do tempo por estado, saturação da direção, desvio-padrão de vref, P/R online (cones confirmados versus vistos uma vez).
- [ ] Atualizar `README_cones.md` e `CONTROLE_STANLEY.md`.

---

## 4. Ordem recomendada e impacto esperado

| Ordem | Item | Esforço | Impacto |
|-------|------|---------|---------|
| 1 | F0 medições + F1 calibração e câmera | 1,5 d | Corrige D6 e a escala lateral. Sem isso, o resto fica mal ajustado. |
| 2 | F2 desempenho | 2 d | 3–4× mais fps, latência 3× menor |
| 3 | F4.1–4.3 mapa + centro cobrindo o eixo | 2 d | Resolve D1, D2 e D5 |
| 4 | F5 pure pursuit | 1,5 d | Resolve D2 e D9 |
| 5 | F6 velocidade | 1 d | Resolve D7 e D8 |
| 6 | F4.4 critérios de uma cor + F3 detecção | 3 d | Curvas fechadas e robustez |

**Quick wins** (aplicados; validar no próximo ensaio):
1. ✅ Trecho distante e **reto** (|κ| < 0,25 1/m) libera a partida até 2,6 m à frente, limitado a 0,35 m/s (`fva_car/cones.py`, `CONE_FAR_START_M`). Num replay do E1, 172 de 186 quadros ficariam válidos (eram 4). Curvas distantes continuam bloqueadas (D1).
2. ✅ `track_width_cm=180` e `cone_spacing_cm=50` em `main.py:cone_settings`, valores medidos nos logs. **Confirmar com trena** (D6).
3. ✅ `curve_speed` sem `abs(heading)` e `speed_filter_s` de 0,3 para 0,6 (D7).
4. ✅ `camera_settings` opcional em `main.py` (`fourcc`, `exposure`, `auto_wb`, `wb_temperature`). Vazio por padrão: os valores dependem da webcam (`v4l2-ctl --list-ctrls`).
5. ✅ `cone_max_lateral_jump_m` de 0,12 para 0,25.

---

## 5. Perguntas em aberto
1. Quais são a largura real do corredor e o espaçamento entre cones? Os logs medem ~1,8 m e ~0,5 m. O código espera 0,9 m e 0,7 m.
2. A câmera pode ser inclinada para baixo, ou a montagem é fixa pela prova?
3. Por que a IMU foi comentada (`1ae7dc0`)? Falha de hardware ou ruído? O giroscópio é importante para o mapa.
4. As cores podem **realmente** trocar de lado no meio da pista, ou isso só acontece entre provas? Se for só entre provas, fixar cor ↔ lado no início simplifica muito.

---

## 6. Visão: o que já foi feito e próximos passos (revisão de 08/11)

**Feito** (resultado medido em 154 quadros reais dos ensaios E1/E2):
- Segmentação: a tabela (a\*, b\*) → matiz/croma Lab substitui `cartToPolar` em float, e `_grow` não usa mais `np.unique`. O resultado é **idêntico** e 23% mais rápido.
- `_bases`: filtro vetorizado das manchas, tabelas de escala por linha e uma única transformada de distância no lugar de 8 `pad+erode`. A saída também é **idêntica** (154/154 quadros), com −19% no quadro 640×480.
- Cones cortados só pela lateral (`partial_cones`) e largura do pé pelo terço inferior: +40 bases verdes, +51 vermelhas, e 22 quadros passaram a ser válidos contra 6 que deixaram de ser (103 → 119 válidos).
- `green_chroma_min` separado do vermelho, para o verde-azulado escuro na sombra (croma 4–12). Fica no padrão 12 porque a faixa larga trouxe ruído nesses ensaios. O ajuste deve ser feito com a câmera em exposição fixa.
- `tools/ajustar_camera.py` (exposição/WB/MJPG) e `benchmark_visao.py --cones`.

**Gargalos restantes**
1. **O laço Python por mancha** ainda consome cerca de 70% da detecção. Quase todas as manchas que chegam ao laço não são cones. Isso se resolve com um classificador (abaixo) e limiares de cor mais estreitos.
2. ✅ **GIL**: a visão agora roda em processo separado (`fva_car/vision_process.py`, `vision_process: True`). Num teste com carga simulada, a detecção ficou em ~11 ms em processo, contra 35 ms em thread. Falta medir no Pi com `benchmark_visao.py --cones` e com o log de um ensaio.
3. **FOV estreito**: com ~60°, os cones a ±0,9 m só aparecem a partir de ~1,6 m. **Uma câmera grande-angular (~120°)** veria os cones ao lado do carro. É a maior alavanca de hardware para o seguimento: basta corrigir a distorção apenas dos pontos de base (`cv2.undistortPoints`, custo desprezível) e recalibrar.
4. **Câmera baixa**: perto do horizonte, 1 px de erro na base a 2,5 m vale cerca de 5 cm. Inclinar a câmera para baixo ou subi-la melhora a precisão métrica.

**Redes neurais no Pi 4 (sem acelerador)**

| Opção | Custo por quadro (estimativa; medir no Pi) | Prós | Contras |
|-------|-------------------------------------------|------|---------|
| **Híbrido: cor gera candidatos + CNN mínima classifica recortes 32×32** (recomendado) | ~0,2 ms por recorte × 10–30 = 2–6 ms | Elimina falsos positivos (roupa, porta, guarda-chuva). Permite afrouxar a cor para pegar o verde escuro. Treina com poucos dados. Mantém a base precisa do pipeline atual. | Exige montar o conjunto de recortes (dá para extrair dos logs). |
| YOLOv8n/YOLO11n 320×320 int8 (NCNN ou ONNX Runtime) | ~70–120 ms, ocupando os 4 núcleos | Muito robusto à luz e ao tom de cor | Latência alta e disputa CPU com o controle. Exige ~300–500 imagens rotuladas com caixa. |
| YOLO no Coral USB Edge TPU | ~10–20 ms | Robusto e rápido | Hardware extra (~US$ 60) e conversão para Edge TPU |
| Sem rede: tabela de cor aprendida (pixels rotulados → LUT 3D) + HOG+SVM nos recortes | ~1–3 ms | Simples e determinístico | Menos robusto que a CNN a mudanças de luz |

Caminho sugerido para o híbrido:
1. Gravar ensaios com `--record-frames 3`.
2. Extrair recortes em volta de cada candidato (aceitos e rejeitados) e revisar os rótulos com `rotular_cones.py`.
3. Treinar uma CNN de 3 classes (cone_verde, cone_vermelho, nao_cone), com ~2 convoluções e entrada de 32×32, no notebook.
4. Exportar para ONNX e rodar com `cv2.dnn` no Pi, com um único lote de recortes por quadro.
5. Medir com `avaliar_cones.py` antes e depois.

**Implementado (08/11):** a estrutura do classificador híbrido.
- `tools/cone_classifier.py`: recorte e inferência com `cv2.dnn`.
- `tools/extrair_recortes_cones.py`: gera o conjunto de treino.
- `tools/treinar_classificador_cones.py`: treina, exporta ONNX e confere o modelo no OpenCV.
- Integração no detector, no processo de visão e no `main.py` (`cone_classifier` e `--classifier`).

Num teste com pseudo-rótulos, a exportação bateu com o PyTorch (diferença de 3e-7) e o custo ficou em 0,02–0,03 ms por recorte no notebook. **Falta gravar e rotular dados reais e treinar.**

## 7. Estimador do corredor (implementado em 08/11)

`fva_car/corridor.py` cobre as fases F4.3 e F4.4 com um único filtro de Kalman
estendido iterado. O estado é `[e, psi, kappa, dkappa, W]` (clotoide), previsto
pela odometria e corrigido pelos cones das duas cores num só ajuste. Os
estados `AMBAS` / `SO_EXTERNA` / `SO_INTERNA` / `UMA_BORDA_RETA` / `MEMORIA` /
`PERDIDO` da tabela 4.4 saem do próprio modelo. Está ligado por padrão
(`cone_estimator`), com `--no-estimator` para voltar. Detalhes e replay estão em
`README_cones.md`.

Replay (`tools/replay_corredor.py`) nos ensaios de 23/10:

| | E1 válidos | E2 válidos | Salto p90 E2 (e / psi) |
|---|---|---|---|
| Antes (log) | 4/186 | 113/246 | 33 cm / 30° |
| Estimador | 186/186 | 215/246 | 0,9 cm / 0,6° |

O que os logs mostraram:
- **Um arco de curvatura constante não basta.** No grampo do E2, os cones verdes
  atravessam a frente a 2,5–3 m. Daí a clotoide (`dkappa`).
- **Carro parado é perigoso para um filtro**: o mesmo cone em vários quadros não é
  informação nova. O ruído por tempo vale só para `e`/`psi`/`W`. Na curvatura, o
  ruído por tempo fazia a reta do E1 "entortar" em 25 s parado.
- **A calibração não cobre a região dos cones.** Os pontos vão de 0,75 a 1,65 m e de
  ±0,3 m, mas os cones ficam a 1,5–3 m e ±0,9 m. No E1 (reta), as bordas medidas
  divergem (largura de 1,74 m a 1,5 m e de 1,82 m a 2,7 m). O estimador então
  dá o carro 0,12–0,15 m à direita do centro, com rumo de −0,04 a −0,07 rad, quando
  o esperado era ~0. **Recalibrar com pontos até 3 m e ±1 m** (fase F1) é o
  próximo passo de maior impacto para a precisão.
- Falta validar no carro: custo no Pi (~1,7 ms no notebook) e ajuste de
  `row_sigma`, `memory_m` e `kappa_noise` com logs novos.

