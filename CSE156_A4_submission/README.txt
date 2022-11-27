***[CSE 156 SP 2022: assignment 4: HMM and Viterbi]***

Make sure all the files are under the same directory

In order to obtain the result in the report, please run the following:

Then should be able to run :
# Each line is for one stat
 > python eval_gene_tagger.py gene.key gene_dev_group.p1.out_trigram
 > python eval_gene_tagger.py gene.key gene_dev.p1.out_trigram
 > python eval_gene_tagger.py gene.key gene_dev_group.p1.out
 > python eval_gene_tagger.py gene.key gene_dev.p1.out
 > python eval_gene_tagger.py gene.train_group  gene_train_group.p2.out
 > python eval_gene_tagger.py gene.train_unk  gene_train.p2.out


# This is to generate the counts, but all the counts has been generated
 > python count_freqs.py gene.train_group > gene.counts_group
 > python count_freqs.py gene.train > gene.counts
 > python count_freqs.py gene.train_unk > gene.train_counts_unk





***[ Files ]***

There are 6 python files in this folder:

- (baseline.py): This file implements baseline model with maximum likelihood probability from emission probability. This file generates gene_dev.p1.out file to be evaluated with gene.key and gene_train.p2.out with gene_train

- (baseline_grouping.py): This file is similar to baseline.py but the rare_words are grouped by different method.  This file generates gene_dev_group.p1.out file to be evaluated with gene.key and gene_train_group.p2.out with gene_train

- (trigram_HMM.py): This file implements trigram_HMM and outputs file gene_dev.p1.out_trigram to be evaluated with gene.key
- (trigram_HMM_grouping.py) This file is similar to trigram_HMM.py but the rare words are grouped by different method. It outputs file gene_dev_group.p1.out to be evaluated with gene.key
- (count_freqs.py): Provided file to get the counts of each word. Helpful for emission probability and transition probability 
- (eval_gene_tagger.py): provided file to evaluate the model by obtaining the precision, recall and f-score 



There are 15 txt files in this folder:
The important files to run are indicated above and the python file that generated the file are also indicated. The name of each file is informative of the identity.

** ACKNOWLEDGE **
[Hwang, S. H.]

