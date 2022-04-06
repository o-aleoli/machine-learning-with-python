## Book Recommendation Engine using KNN

Nesse projeto eu implementei um sistema de recomendações com o algoritmo de K-vizinhos mais próximos e as notas dadas para os livros no banco de dados [Book-Crossings ](http://www2.informatik.uni-freiburg.de/~cziegler/BX/).
A função `get_recommends()` tem como entrada o nome da obra e como saída o nome e semelhança, de zero a um, de outras cinco obras.
Primeiro o banco de dados foi processado com o objetivo de remover livros pouco populares e usuários pouco ativos e depois transformado numa tabela esparsa, fazendo com que cada obra vire um vetor n-dimensional com a nota dada por todos os usuários.
Enfim, a distância entre obras foi obtida calculando o cosseno entre vetores com a função `NearestNeighbors` do pacote `scikit-learn`.

### Fonte do Banco de Dados

Improving Recommendation Lists Through Topic Diversification,
Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, Georg Lausen; Proceedings of the 14th International World Wide Web Conference (WWW '05), May 10-14, 2005, Chiba, Japan. To appear.
