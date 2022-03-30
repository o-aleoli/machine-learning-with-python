# Cat and Dog Image Classifier

Esse projeto consiste em corretamente classificar imagens de cães e gatos pelo menos 63% das vezes.
Para isso, construí uma rede neural covolucional com 128 filtros quadrados de 10 pixels, passando por uma camada de covolução abstraindo os 128 filtros em 32 terminando em duas camadas neurais densas com 64 e 1 neurônios cada.
Mais detalhes em `fcc_cat_dog.ipynb`.

Após treino em 15 epochs de 2000 figuras, o modelo foi capaz de reconhecer pets em 70% das figuras de validação.
