# Documentação do Projeto de Pipeline de Dados e Modelo Preditivo

## Visão Geral
Este projeto consiste na criação de um pipeline de dados que inclui a extração, preparação de dados, treinamento de um modelo preditivo de árvore de decisão, e a realização de previsões com base em novos dados. O objetivo é demonstrar a implementação de um processo de aprendizado de máquina desde a extração de dados até a inferência.

## Estrutura do Projeto
O projeto é composto por dois scripts principais:
- `train.py`
- `predict.py`

### 1. `train.py`
Este script é responsável pelo treinamento do modelo. Ele segue as etapas:
- **Extração de Dados**: Utiliza o conjunto de dados `Iris` da biblioteca `scikit-learn`.
- **Preparação de Features**: Converte os dados em um DataFrame `pandas` para separação entre features (X) e rótulos (y).
- **Treinamento do Modelo**: Treina um classificador `DecisionTreeClassifier` com profundidade máxima de 2.
- **Serialização do Modelo**: Salva o modelo treinado em um arquivo `trained_classifier.pkl`.

#### Estrutura do Código
```python
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Extração de Dados
def extract_data():
    data = load_iris()
    return data

# Preparação de Features
def preparing_features(data):
    x = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=['target'])
    return x, y

# Treinamento do Modelo
def train_model(x, y):
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(x, y)
    return model

# Serialização
def serialize_object(model):
    with open("trained_classifier.pkl", "wb") as file:
        pickle.dump(model, file)

# Execução do Pipeline
def run():
    data = extract_data()
    x, y = preparing_features(data=data)
    model = train_model(x, y)
    serialize_object(model=model)

if __name__ == "__main__":
    run()
```

### 2. `predict.py`
Este script realiza previsões usando o modelo previamente treinado:
- **Carregamento de Dados**: Cria um `DataFrame` com novos dados para previsão.
- **Carregamento do Modelo**: Carrega o modelo salvo no arquivo `trained_classifier.pkl`.
- **Predição**: Realiza previsões com base nos dados fornecidos.
- **Saída de Resultados**: Imprime os resultados das predições.

#### Estrutura do Código
```python
import pickle
import pandas as pd

# Carregamento de Dados
def load_data():
    new_data = pd.DataFrame([[1, 1.5, 5, 6]])
    return new_data

# Carregamento do Modelo
def load_model():
    with open("trained_classifier.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Realização de Predições
def make_predictions(data, model):
    return model.predict(data)

# Saída de Resultados
def write_results(predictions):
    print(predictions)

# Execução
def run():
    new_data = load_data()
    model = load_model()
    predictions = make_predictions(data=new_data, model=model)
    write_results(predictions=predictions)

if __name__ == "__main__":
    run()
```

## Como Executar
1. **Treinamento do Modelo**:
   - Execute o script `train.py` para treinar o modelo.
   ```bash
   python train.py
   ```
   Isso gerará o arquivo `trained_classifier.pkl` contendo o modelo treinado.

2. **Previsão**:
   - Execute o script `predict.py` para realizar previsões com o modelo treinado.
   ```bash
   python predict.py
   ```
   O script irá imprimir os resultados das previsões no console.

## Requisitos
- `Python 3.x`
- Bibliotecas: `pandas`, `numpy`, `scikit-learn`, `pickle`

## Conclusão
Este projeto é um exemplo prático de um pipeline de dados e modelo preditivo que pode ser expandido para incluir conjuntos de dados maiores, diferentes técnicas de machine learning e mais funcionalidades conforme a necessidade.

