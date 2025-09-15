# NLP-with-Disaster-Tweets
NLP Classification to classify tweets as disaster/non-disaster

This repository contains my submission to classify tweets as disaster/non-disaster for the following Kaggle competition: https://www.kaggle.com/competitions/nlp-getting-started/overview

This competition submissions were evaluated on the basis of f1-score and the following code achieved a score of 0.82.

For the dataset, DistilBERT model was trained to classify the tweets accurately. Further, the model was hyperparameter tuned with 5-fold cross validation. To speed up the runtime, I used 2 T4 GPUs and parallelized the experiments across the two GPUs. The best hyperparameter combination was later used to generate the submission file
