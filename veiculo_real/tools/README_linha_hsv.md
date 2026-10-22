# Camera USB: faixa amarela em retas e curvas

Na raiz do projeto:

```bash
venv/bin/python veiculo_real/tools/detectar_linha_amarela_hsv.py
```

A interface usa **Tkinter + Pillow**, portanto funciona com o OpenCV headless
que ja esta instalado no projeto. Nao precisa trocar esse pacote por outra
distribuicao do OpenCV. Se faltar Pillow:

```bash
venv/bin/python -m pip install pillow
```

Para preparar um ambiente separado:

```bash
python3 -m venv ~/venvs/camera-hsv
source ~/venvs/camera-hsv/bin/activate
python -m pip install numpy pillow opencv-python-headless
python veiculo_real/tools/detectar_linha_amarela_hsv.py
```

O Python precisa de Tkinter e de uma sessao grafica. No Ubuntu/Debian, se o
modulo `tkinter` estiver ausente, instale `python3-tk` pelo gerenciador do
sistema. Mantenha `yellow_path.py` na mesma pasta do programa principal.

## Como usar

1. Escolha o dispositivo no seletor **Camera** e clique **Conectar / trocar**.
   No Linux, a lista mostra `/dev/videoN` e o nome do dispositivo. Algumas
   webcams oferecem varios endpoints: escolha outro se o primeiro nao der imagem.
   Em outros sistemas, experimente os indices da lista. O campo tambem aceita
   um indice digitado ou um caminho `/dev/videoN`. **Atualizar lista** procura
   dispositivos conectados depois de abrir o programa.
2. Posicione a camera olhando para a pista, com o trecho proximo da faixa na
   parte inferior da imagem. O detector procura uma faixa longitudinal que
   avance pela imagem. Ele acompanha retas, arcos e curvas em S.
3. Se houver duas faixas, clique perto da base da desejada nas visualizacoes
   **Destaque** ou **Original**. O clique muda a preferencia horizontal de
   selecao; nao obriga o detector a aceitar um trecho sem evidencia.
4. Use **Caminho** para ver apenas a faixa selecionada e **HSV** para conferir
   a segmentacao forte antes da selecao geometrica. A visualizacao HSV pode
   mostrar outros objetos amarelos; eles nao sao todos selecionados como caminho.
5. Verde representa suporte de cor, inclusive tinta desbotada aceita pelo
   limiar tolerante. **Azul conecta tracos separados de uma faixa pontilhada**;
   laranja representa pequenos buracos restantes na faixa.
   A linha branca marca o centro da faixa. A porcentagem de suporte mede pixels
   com evidencia de cor dentro do caminho, nao uma probabilidade de acerto.
6. Na aba **Gaps**, **Conectar faixas pontilhadas** vem ligado. Ajuste o
   **Gap pontilhado maximo**, inicialmente 120 px. O valor usa a escala de uma
   imagem com 480 px de altura. A conexao so acontece entre observacoes atuais,
   com posicao, largura e tangentes compativeis dos dois lados. O limite extra
   complementa o ajuste **Conectar falhas** da aba Caminho.

## Ajustes principais

| Ajuste | Efeito |
|---|---|
| Conectar falhas | Distancia vertical maxima entre observacoes; comece em 36 px e experimente 45–60 para tinta interrompida. O valor e escalado para altura de 480 px. Valores altos podem unir trechos diferentes. |
| Conectar faixas pontilhadas | Ativa a ligacao azul entre tracos separados, inclusive em curvas. Desligar restaura apenas a recuperacao de falhas curtas. |
| Gap pontilhado maximo | Limite extra para os espacos do pontilhado, inicialmente 120 px. Aumente aos poucos para gaps maiores; conexoes com mudancas abruptas de orientacao sao rejeitadas. |
| Largura maxima | Limita a largura de cada trecho como porcentagem da imagem. Reduza para rejeitar manchas/objetos largos; aumente se a faixa estiver muito perto da camera. |
| Extensao minima | Exige comprimento vertical minimo. Aumente para rejeitar objetos curtos; reduza se so um trecho pequeno da pista estiver visivel. |
| Ignorar topo | Exclui a parte superior, onde frequentemente aparecem objetos fora da pista. |
| Estabilidade temporal | Suaviza a geometria entre frames recentes, somente quando ha deteccao atual compativel. Reduza se a camera se mover rapidamente. |
| H minimo / maximo | Faixa de matiz, inicialmente 18–40 na escala de 0–179 do OpenCV. |
| S minimo | Padrao 25, na escala 0..255, conforme o ajuste da linha real. Aumentar ajuda a excluir superficies esbranquicadas. |
| V minimo | Diminua para sombras e manchas escuras. |
| CLAHE | Compensa contraste no brilho sem mudar H e S. Compare ligado/desligado em sua pista. |

**Rebuscar no centro** limpa a memoria e a preferencia de selecao. **Restaurar
todos os ajustes** volta aos padroes. **Esc** fecha a janela e libera a captura.
Tambem e possivel definir a camera e resolucao iniciais por argumentos:

```bash
venv/bin/python veiculo_real/tools/detectar_linha_amarela_hsv.py --camera 1 --width 1280 --height 720
```

O processamento usa no maximo 640 px de largura, preservando a proporcao.
Captura/abertura de camera ocorre em uma thread separada; o seletor continua
disponivel enquanto o driver abre o dispositivo. A aplicacao processa o frame
mais recente, evitando uma fila crescente de imagens atrasadas.

## Como a selecao funciona

O HSV produz evidencia forte e tolerante de amarelo. Um fechamento morfologico
pequeno tapa buracos locais. A imagem e dividida em bandas horizontais e uma
busca por programacao dinamica escolhe uma cadeia de trechos compativeis em
posicao, largura e direcao, a partir da parte inferior. So sao aceitas cadeias
com extensao, proporcao alongada e quantidade suficientes de evidencia forte.
Um polinomio cubico robusto suaviza o centro e reconstrui apenas o intervalo
entre observacoes, permitindo pequenas lacunas e curvas. O corredor final
precisa ter pelo menos 50% de suporte de cor tolerante e 12% de suporte forte,
para nao transformar uma sequencia de pontos aleatorios em uma faixa.
Quando a cadeia possui gaps longos de pontilhado validados, o suporte tolerante
minimo passa para 28%, mantendo 12% de suporte forte. Cada ligacao longa exige
trechos com comprimento suficiente para medir a tangente nos dois lados,
variacao angular de ate 35 graus e alinhamento com a direcao media. Uma simples
mudanca lateral entre dois objetos paralelos nao e suficiente para conecta-los.

O `PathResult.path_mask` e a linha central `PathResult.center` incluem os gaps
aceitos, proporcionando um caminho continuo para um consumidor de percepcao.
`PathResult.dash_mask` identifica especificamente a interpolacao azul, e
`observed_mask` permite distinguir a evidencia de cor da estimativa geometrica.
O programa USB continua sendo uma ferramenta de percepcao/visualizacao; nao foi
conectado aos comandos de direcao ou velocidade do carrinho.

Isso filtra manchas compactas e objetos largos, mas e uma heuristica geometrica,
nao um classificador semantico: um objeto amarelo fino, longo e alinhado com a
pista ainda pode parecer uma faixa. Reflexos estourados, oclusoes extensas,
cruzamentos e retornos que se sobrepoem na mesma altura da imagem podem causar
perda ou ambiguidade. Sem evidencia atual, o programa deixa de desenhar o caminho;
nao prolonga uma linha antiga indefinidamente. Ele nao envia comandos ao carrinho.

## Verificacao sem camera

```bash
venv/bin/python -m unittest discover -s veiculo_real/tools/tests -p 'test_yellow_path.py' -v
```

As regressoes cobrem retas/curvas, falhas curtas e longas, pontilhados retos e
curvos com preenchimento azul, rejeicao de tracos desalinhados, objetos fora
da faixa, tinta desbotada, selecao entre duas faixas, perda de deteccao e troca
de camera com captura simulada. A webcam e o aspecto da janela devem ser
conferidos em uma sessao grafica real.
