import numpy as np
from collections import defaultdict

f = open("gene.counts", "r")
counts = []
lines = f.readlines()
rare_words = []
for line in lines:
    counts.append(line)
    temp = line.strip("\n").split(" ")
    if int(temp[0]) < 5:
        rare_words.append(temp[-1])
f.close()


counts_file = "gene.counts"
words = {}
ngrams = {1 : {}, 2 : {}, 3 : {}}
word_counts = {}
for l in open(counts_file):
    t = l.strip().split()
    count = int(t[0])
    key = tuple(t[2:])
    if t[1] == "1-GRAM": ngrams[1][key[0]] = count
    elif t[1] == "2-GRAM": ngrams[2][key] = count
    elif t[1] == "3-GRAM": ngrams[3][key] = count
    elif t[1] == "WORDTAG":
        words[key] = count
        word_counts.setdefault(key[1], 0)
        word_counts[key[1]] += count


def replace_word(word):
    "Returns the word or its replacement."
    if word_counts[word] < 5: return "_RARE_"
    else: return word

f = open("gene.train", "r")
f2 = open("gene.train_unk", "w")
counts = []
lines = f.readlines()
for line in lines:
    if line != "\n":
        temp = line.strip("\n").split(" ")
        word = replace_word(temp[0])
        temp[0] = word
        f2.write(" ".join(temp) + "\n")
    else:
        f2.write("\n")
f.close()
f2.close()

counts_file = "gene.train_counts_unk"
words = {}
ngrams = {1 : {}, 2 : {}, 3 : {}}
word_counts = {}
lines = []
for l in open(counts_file):
    lines.append(l)
    t = l.strip().split()
    count = int(t[0])
    key = tuple(t[2:])
    if t[1] == "1-GRAM": ngrams[1][key[0]] = count
    elif t[1] == "2-GRAM": ngrams[2][key] = count
    elif t[1] == "3-GRAM": ngrams[3][key] = count
    elif t[1] == "WORDTAG":
        words[key] = count
        word_counts.setdefault(key[1], 0)
        word_counts[key[1]] += count

total_O = 345128
total_gene = 41072
tags = ["O", "I-GENE"]


def _emission_baseline(counts=lines):
    emission = defaultdict()
    for w in counts:
        temp = w.strip("\n").split()
        if temp[-2] == "O":
            emission[(temp[-1], temp[-2])] = int(temp[0]) / total_O
        if temp[-2] == "I-GENE":
            emission[(temp[-1], temp[-2])] = int(temp[0]) / total_gene
    return emission

emission = _emission_baseline()

def _tagger(sentence):
    tags = []
    for w in sentence:
        O = None
        I_GENE = None
        if w in word_counts:
            try:
                O = emission[(w,"O")]
                I_GENE = emission[(w,"I-GENE")]
            except:
                if O == None:
                    O = 0
                if I_GENE == None:
                    I_GENE = 0
        else:
            O = emission[("_RARE_","O")]
            I_GENE = emission[("_RARE_","I-GENE")]
        if O > I_GENE:
            tags.append("O")
        else:
            tags.append("I-GENE")
    return tags
try:
    f = open("gene_dev.p1.out", "x")
except:
    f = open("gene_dev.p1.out", "w")
f2 = open("gene.dev","r")
lines = f2.readlines()
cur = []
for line in lines:
    if line == "\n":
        output = _tagger(cur)
        for idx,ele in enumerate(output):
            f.write(cur[idx] + " " + ele + "\n")
        f.write("\n")
        cur = []
    else:
        line = line.strip("\n")
        cur.append(line)

try:
    f = open("gene_train.p2.out", "x")
except:
    f = open("gene_train.p2.out", "w")

f2 = open("gene.train_no_label", "r")
lines = f2.readlines()
cur = []

for line in lines:
    if line == "\n":
        output = _tagger(cur)
        for idx, ele in enumerate(output):
            f.write(cur[idx] + " " + ele + "\n")
        f.write("\n")
        cur = []

    else:
        line = line.strip("\n")
        cur.append(line)
f.close()
f2.close()