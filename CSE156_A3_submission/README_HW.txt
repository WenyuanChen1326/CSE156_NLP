***[CSE 156 SP 2022: assignment 3: Comparing Language Models ]***

You may need to first run:
 > pip install tabulate

Then should be able to run :
 > python data.py
 > python data_adaptation,py

***[ Files ]***

There are 6 python files in this folder:
- (lm.py): This file describes the higher level interface for a language model, and contains functions to train, query, and evaluate it. An implementation of a simple back-off based unigram model is also included, that implements all of the functions
of the interface.
- (generator.py): This file contains a simple word and sentence sampler for unigram model. Since it supports arbitarily complex language models, it is not very efficient. 

-  (data.py): The primary file to run. This file contains methods to read the appropriate data files from the archive, train and evaluate all the trigram language models (by calling “lm.py” and trigram_laplace.py), and generate sample sentences from all the models (by calling  “generator.py” and "generator_tri.py"). It also saves the result tables into LaTeX files. The Laplace delta value is also tuned. 

- (trigram_laplace.py): This file describes the higher level interface for a language model, and contains functions to train, query, and evaluate it. An implementation of a trigram model with Laplace smoothing is also included, that implements all of the functions
of the interface.

- (generator_tri.py): This file contains a simple word and sentence sampler for trigram with laplace model. Since it supports arbitarily complex language models, it is not very efficient. The prefix set with the start tokens (*, **). 

-  (data_adaptation.py): The secondary file to run. This file contains methods to read the appropriate data files from the archive, train and evaluate all the unigram language models (by calling “lm.py”), and generate sample sentences from all the models (by calling  “generator.py”). It also saves the result tables into LaTeX files. The training corpus for different domains are combined to see the adaptation effect. 

