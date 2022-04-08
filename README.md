# Machine Learning With Python course's projects

These are my projects from the FreeCodeCamp's Machine Learning With Python course.
In general, they use TensorFlow and Keras packages for machine learning models, and Pandas for ETL and statistical analysis.
The details on each project's planning, development, and troubleshooting are availble on Jupyter Notebooks (in PT-BR) at their respective folders.
Original files and problem descriptions are availble on [author's website (FreeCodeCamp)](https://www.freecodecamp.org/learn/machine-learning-with-python/#machine-learning-with-python-projects).

## Rock Paper Scissors

On the course's first project, the player must win at least 60% of 1000 plays against four artificial intelligences.
In my approach, I first trained a recurrent neural network to correctly classify the opponent using its first 200 plays and then choose the correct strategy against it.
This approach were capable of winning the AI in 87% ~ 96% of the plays at the unit test.
More details on `rock-paper-scissors/resolucao_rps.ipynb` (PT-BR).

## Cat and Dog Image Classifier

The project proposes building a covolutional neural network to correctly classify at least 63% of cats and dogs figures.
The CNN was built with a layer of 128 square filters with 10 pixels each, an pooling layer with 32 filters and two dense layers with 64 and 1 neurons each.
The model were capable of identifying cats and dogs at 70% of the validation pictures after training for 15 epochs with 200 pictures each.
More details on `fcc_cat_dog.ipynb`.

## Book Recommendation Engine using KNN

In this project I built a recommendation system with the K-Nearest Neighbors algorithm and ratings from the [Book-Crossings database](http://www2.informatik.uni-freiburg.de/~cziegler/BX/).
By inserting the book title in `get_recommends()` function it returns the 5 closest books and their simmilarities to the input from 0 to 1.

## Linear Regression Health Costs Calculator

The project proposes building an regression model to predict health care expenses from a database containing age, gender, body mass index (BMI) and region data.
It is also required to train the model until achieves a mean absolute error smaller than US$ 3500.
