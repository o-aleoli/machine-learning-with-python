# Rock Paper Scissors

O primeiro projeto do curso consiste em derrotar quatro inteligências artificiais (IA) em pelo menos 60% das 1000 jogadas.
Minha abordagem consiste em desenvolver uma rede neural recorrente para reconhecer qual oponente está jogando apenas pelas suas 200 primeiras jogadas e, a partir dessa informação, adotar a estratégia mais adequada.
Mais detalhes em `resolucao_rps.ipynb`.

O projeto teve ótimos resultados no unit test, derrotando as IAs em 87% ~ 96% das partidas.

Os arquivos `generate_ETL_plays.py` e `generate_ETL_plays_validation.py` contém o código para gerar os arquivos `db_train.csv`, `db_test.csv` e `db_validation.csv` usados no desenvolvimento da rede neural.
A rede neural foi treinada, testada e validada nos arquivos `rnn_train_module.py`, `rnn_test_module.py` e `rnn_validation_module.py`, respectivamente.
Os pesos usados pela rede neural foram salvos no arquivo `trained_model.h5`.
A solução completa foi implementada no arquivo `RPS.py`, que se integra ao boilerplate na plataforma do [Replit](replit.com).
