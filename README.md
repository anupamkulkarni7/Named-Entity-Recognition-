# Named Entity Recognition
---

This repo is a NER tagger trained on the CONLL 2003 English dataset and 
pre-trained 50 dimensional GloVE vectors. It is a light, word level system which 
used custom built tokenizers for word-level encoding as well as a 
case-encoding to add character level information. 

Additionally, I have also included an example of a web-based micro-service
developed using *flask* where the trained model can be used to do NER
tagging on input sentences. 

### Usage
#### Training
```
python run.py train --train-src=<training file path> --val-src=<validation file path> --glove-file=<GloVE file path>
```

The details of **run.py** usage can be found by
```
python run.py -h/--help
```

#### Prediction

You can either use the model instance after training, or after loading a
saved model. In case of a saved model, the typical usage will be as 
follows:

```
ner_tagger = NERTagger()
ner_tagger.load()
ner_tagger.predict(input_txt)

```


### Model Performance
To evaluate the model performance on the CONLL test set:
```
python run.py test --test-src=<testing file path>
```
The model achieves an average F1 score of 0.84 on the CONLL 
test set. The detailed results are shown below. 
```
            precision    recall  f1-score   support

      PER       0.90      0.90      0.90      1617
      ORG       0.79      0.81      0.80      1661
      LOC       0.85      0.89      0.87      1668
     MISC       0.70      0.79      0.74       702

micro avg       0.83      0.86      0.84      5648
macro avg       0.83      0.86      0.84      5648

```






