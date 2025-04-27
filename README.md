# Modelo de Sequência de Rede Neural com Atenção

Este projeto implementa um modelo sequência-para-sequência (seq2seq) com mecanismo de atenção usando PyTorch. O modelo consiste em uma arquitetura codificador-decodificador com células GRU (Gated Recurrent Unit) e mecanismo de atenção para melhorar o processamento de sequências. O sistema é projetado para aprender padrões em sequências e gerar sequências de saída correspondentes, com foco na imputação de dados genéticos.

## Índice

- [Instalação](#instalação)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Uso](#uso)
  - [Script Principal: impute.py](#script-principal-imputepy)
  - [Formato dos Dados de Entrada](#formato-dos-dados-de-entrada)
  - [Parâmetros](#parâmetros)
  - [Exemplos de Comandos](#exemplos-de-comandos)
- [Processamento de Arquivos VCF](#processamento-de-arquivos-vcf)
- [Arquivos de Saída](#arquivos-de-saída)
- [Arquitetura do Modelo](#arquitetura-do-modelo)
- [Solução de Problemas](#solução-de-problemas)

## Instalação

### Pré-requisitos

- Python 3.7 ou superior
- GPU compatível com CUDA (recomendado, mas não obrigatório)

### Dependências

Instale os pacotes necessários usando pip3:

```bash
pip3 install torch numpy pandas matplotlib scikit-learn tqdm
```

Para aceleração com GPU (recomendado para treinamento mais rápido):

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Substitua `cu118` pela sua versão do CUDA, se for diferente.

### Configuração

Crie um diretório de log para armazenar arquivos de saída:

```bash
mkdir -p log
```

Este diretório é necessário para armazenar arquivos de log e resultados de avaliação.

## Estrutura do Projeto

```
project/
├── log                # Armazenar arquivos de log e resultados de avaliação
├── attnDecoder.py     # Implementação do decodificador com atenção
├── encoder.py         # Implementação do codificador
├── train.py           # Loops de treinamento e utilitários
├── evaluation.py      # Funções de avaliação do modelo
├── helpers.py         # Funções utilitárias
├── impute.py          # Manipulação e processamento de dados (ponto de entrada principal)
├── tensorHelpers.py   # Utilitários de manipulação de tensores
├── device.py          # Seleção de dispositivo (CPU/GPU)
├── logger.py          # Configuração de log
├── plot.py            # Utilitários de visualização
└── vcftogen.py        # Utilitário de processamento de arquivos VCF
```

## Uso

### Script Principal: impute.py

O ponto de entrada principal para o projeto é o `impute.py`. Este script lida com o carregamento de dados, criação do modelo, treinamento e avaliação.

Uso básico:

```bash
python3 impute.py -input seu_arquivo_de_entrada.txt
```

Ou, caso queira utilizar o arquivo padrão:

```bash
python3 impute.py
```

### Formato dos Dados de Entrada

O arquivo de entrada deve conter dados de sequência com uma sequência por linha. Cada linha representa uma amostra, e cada caractere na linha representa um marcador genético (tipicamente o alelo que aparece com maior frequência na população é substituído pelo número zero, e o alelo que aparece com menor frequência na população [muitas vezes indicando a variante com mutação] é substituído pelo número um).

Exemplo de formato de arquivo de entrada:
```
0010110001001010101...
1001010101010010101...
0101010101001010101...
```

O modelo irá:
1. Mascarar uma parte dos marcadores (determinada por `-rel_mask`)
2. Treinar para prever os valores mascarados com base nos marcadores visíveis
3. Avaliar a precisão da previsão em um conjunto de teste

### Parâmetros

| Parâmetro | Descrição | Padrão | Tipo |
|-----------|-----------|--------|------|
| `-hidden_size` | Número de unidades ocultas no modelo | 512 | int |
| `-lrate` | Taxa de aprendizado para otimização | 0.01 | float |
| `-it` | Número de iterações de treinamento | 50000 | int |
| `-eval_ev` | Avaliar modelo a cada N iterações | 1000 | int |
| `-verbose` | Nível de detalhamento (0: Nenhum, 1: Info, 2: Debug) | 0 | int |
| `-input` | Caminho do arquivo de entrada | "result_aux_big.txt" | str |
| `-rel_mask` | Número relativo de variantes mascaradas (0-1) | 0.3 | float |
| `-rel_test` | Tamanho relativo do conjunto de teste (0-1) | 0.2 | float |
| `-max_length` | Comprimento máximo da sequência | 100 | int |
| `-length` | Comprimento do conjunto de dados a ser usado | 1000 | int |
| `-offset` | Deslocamento da posição original do conjunto de dados | 0 | int |
| `-dropout` | Taxa de dropout para regularização | 0.01 | float |
| `-ilang_size` | Tamanho do bloco da linguagem de entrada | 10 | int |
| `-olang_size` | Tamanho do bloco da linguagem de saída | 5 | int |

### Exemplos de Comandos

**Treinamento básico com parâmetros padrão:**
```bash
python3 impute.py -input data/my_sequence_data.txt
```

**Treinamento com parâmetros personalizados:**
```bash
python3 impute.py -input data/my_sequence_data.txt -hidden_size 256 -lrate 0.001 -it 10000 -verbose 1
```

**Treinamento com alta verbosidade e frequência de avaliação personalizada:**
```bash
python3 impute.py -input data/my_sequence_data.txt -verbose 2 -eval_ev 500
```

**Usando um arquivo de configuração:**
Você também pode usar um arquivo de configuração com parâmetros:
```bash
python3 impute.py @config.txt
```

Onde `config.txt` contém parâmetros como:
```
-hidden_size 256
-lrate 0.001
-it 10000
```

## Processamento de Arquivos VCF

O projeto inclui um script utilitário `vcftogen.py` para processamento de arquivos VCF (Variant Call Format), que são comumente usados em análise de dados genéticos.

Uso:
```bash
python3 vcftogen.py
```

Este script:
1. Lê um arquivo VCF (padrão: 'chr1.vcf')
2. Processa 20.000 linhas do arquivo
3. Realiza verificação de taxa de chamada
4. Gera dados processados para "output.txt"

Nota: Pode ser necessário modificar os caminhos de arquivo codificados no script para seu caso de uso específico.

## Arquivos de Saída

O modelo gera vários arquivos de saída durante o treinamento e avaliação:

1. **Arquivos de Log**: Criados no diretório `log/` com carimbos de data/hora
   - Progresso do treinamento
   - Métricas de avaliação
   - Formato: `log/{timestamp}.log`

2. **Arquivos de Precisão**: Arquivos CSV com métricas detalhadas de precisão
   - Formato: `log/Acc_{timestamp}.csv`
   - Contém detalhamento de precisão por frequência de variante (comum, incomum, rara)
   - As colunas incluem:
     * Acc: Precisão geral
     * common positives/acertos/acc: Métricas para variantes comuns
     * uncommon positives/acertos/acc: Métricas para variantes incomuns
     * rare positives/acertos/acc: Métricas para variantes raras

3. **Saída do Console**: Durante o treinamento, você verá:
   - Atualizações de progresso em intervalos especificados por `-eval_ev`
   - Tempo decorrido
   - Iteração atual
   - Valor de perda
   - Métricas de precisão
   - Previsões de amostra (se verbose > 1)

Exemplo de saída do console:
```
2m 13s (1000 2%) LOSS 2.1234 ACC 0.7890
4m 26s (2000 4%) LOSS 1.8765 ACC 0.8123
...
```

## Arquitetura do Modelo

O modelo usa uma arquitetura sequência-para-sequência com atenção:

1. **Codificador**: Classe `EncoderRNN` em `encoder.py`
   - Camada de embedding para processamento de entrada
   - GRU para processamento de sequência

2. **Decodificador com Atenção**: Classe `AttnDecoderRNN` em `attnDecoder.py`
   - Mecanismo de atenção para focar em partes relevantes da sequência de entrada
   - Decodificação baseada em GRU das sequências codificadas

3. **Processo de Treinamento**: Implementado em `train.py`
   - Treinamento iterativo com número especificado de iterações
   - Otimização SGD com taxa de aprendizado configurável
   - Teacher forcing opcional

4. **Avaliação**: Funções em `evaluation.py`
   - Avaliação de amostras aleatórias
   - Avaliação abrangente com métricas de precisão

## Solução de Problemas

### Problemas Comuns

1. **FileNotFoundError: [Errno 2] No such file or directory: 'log/...'**
   - Solução: Crie o diretório de log com `mkdir -p log`

2. **CUDA out of memory**
   - Solução: Reduza o `hidden_size` ou use um comprimento menor do conjunto de dados

3. **Erros de formato do arquivo de entrada**
   - Solução: Certifique-se de que seu arquivo de entrada contenha sequências de comprimento consistente
   - Verifique se o parâmetro `-length` corresponde aos seus dados

4. **Baixa precisão**
   - Tente aumentar o número de iterações (`-it`)
   - Ajuste a taxa de aprendizado (`-lrate`)
   - Aumente o tamanho oculto (`-hidden_size`)
   - Ajuste a proporção de mascaramento (`-rel_mask`)

5. **Treinamento lento**
   - Ative a aceleração de GPU se disponível
   - Reduza o tamanho do conjunto de dados para experimentação mais rápida
   - Diminua a frequência de avaliação (`-eval_ev`)

### Obtendo Ajuda

Se você encontrar problemas não abordados nesta seção de solução de problemas, verifique:
- Os comentários do código para informações adicionais
- Os arquivos de documentação do projeto (`project_documentation.md` e `code_summary.md`)