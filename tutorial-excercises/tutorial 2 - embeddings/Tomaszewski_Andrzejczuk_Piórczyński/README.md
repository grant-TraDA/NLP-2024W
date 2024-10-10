| Method name                      | Model                                | Accuracy | Precision | Recall  | F1 Score |
|----------------------------------|--------------------------------------|----------|-----------|---------|----------|
| Word2Vec                         | LogisticRegression                   | 0.81320  | 0.811159  | 0.81648 | 0.813811 |
| FastText                         | LogisticRegression                   | 0.80008  | 0.799362  | 0.80128 | 0.800320 |
| Word2Vec                         | RandomForest                         | 0.75792  | 0.764653  | 0.74520 | 0.754801 |
| FastText                         | RandomForest                         | 0.71212  | 0.717747  | 0.69920 | 0.708352 |
| Word2Vec                         | SVM                                  | 0.81176  | 0.807771  | 0.81824 | 0.812972 |
| FastText                         | SVM                                  | 0.80052  | 0.796277  | 0.80768 | 0.801938 |
| Word2Vec                         | KNN                                  | 0.69340  | 0.721606  | 0.62976 | 0.672562 |
| FastText                         | KNN                                  | 0.62640  | 0.641882  | 0.57184 | 0.604840 |
| Word2Vec                         | NaiveBayes                           | 0.62756  | 0.620695  | 0.65600 | 0.637859 |
| FastText                         | NaiveBayes                           | 0.54268  | 0.532596  | 0.69736 | 0.603942 |
| Word2Vec                         | XGBoost                              | 0.78232  | 0.785287  | 0.77712 | 0.781182 |
| FastText                         | XGBoost                              | 0.74020  | 0.742548  | 0.73536 | 0.738936 |
| bert-base-uncased + mean pooling | LogisticRegression                   | 0.83760  | 0.841922  | 0.83128 | 0.836567 |
| bert-base-uncased + cls token    | LogisticRegression                   | 0.80924  | 0.817756  | 0.79584 | 0.806649 |
| bert-base-uncased + mean pooling | kNN                                  | 0.74832  | 0.800659  | 0.66128 | 0.724325 |
| bert-base-uncased + cls token    | kNN                                  | 0.65368  | 0.677542  | 0.58648 | 0.628731 |
| bert-base-uncased + mean pooling | XGBoost                              | 0.81032  | 0.813075  | 0.80592 | 0.809482 |
| bert-base-uncased + cls token    | XGBoost                              | 0.75376  | 0.759362  | 0.74296 | 0.751072 |
| fine-tuing                               | huawei-noah/TinyBERT_General_4L_312D | -        | 0.898436  | 0.87752 | 0.887855 |
| fine-tuing                               | distilbert-base-uncased              | -        | 0.936231  | 0.91848 | 0.927271 |
| fine-tuing                               | Movie Review Classifier - Pretrained | -        | 0.927329  | 0.93816 | 0.932713 |
