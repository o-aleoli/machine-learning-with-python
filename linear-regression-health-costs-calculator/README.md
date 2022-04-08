## Linear Regression Health Costs Calculator

Nesse projeto, construo um modelo de regressão que usa dados de idade, gênero, índice de massa corpórea (BMI em inglês), tabagismo e região de moradia para prever o quanto que uma pessoa gastará de plano de saúde.
É requerido que o modelo treinado possua um erro absoluto médio inferior à US$ 3500.

O pré-processamento consistiu em remover outliers, normalizar variáveis numéricas usando Z-score e converter variáveis categóricas em lógicas.
Construí um modelo de rede neural com duas camadas profundas, cada uma com 8 unidades e função de ativação ReLU.
Como se requeriu uma precisão arbitrária, também implementei uma classe de truncamento de treino, o `CallbackEarlyStopMAE()`.
