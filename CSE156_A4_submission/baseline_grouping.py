import numpy as np
from collections import defaultdict

counts_file = "gene.counts"
words = {}
ngrams = {1: {}, 2: {}, 3: {}}
word_counts = {}
for l in open(counts_file):
    t = l.strip().split()
    count = int(t[0])
    key = tuple(t[2:])
    if t[1] == "1-GRAM":
        ngrams[1][key[0]] = count
    elif t[1] == "2-GRAM":
        ngrams[2][key] = count
    elif t[1] == "3-GRAM":
        ngrams[3][key] = count
    elif t[1] == "WORDTAG":
        words[key] = count
        word_counts.setdefault(key[1], 0)
        word_counts[key[1]] += count

def group_word(word, exist=True):
    import re
    if exist:
        if word_counts[word] < 5:
            if not re.search(r'\w', word):
                return '_PUNCS_'
            elif re.search(r'[A-Z]', word):
                return '_CAP_'
            elif word.isupper():
                return "_ALLCAP_"
            elif re.search(r'\d', word):
                return '_NUM_'
            elif re.search(r'(ses\b|ion\b|ty\b|ics\b|ment\b|ence\b|ance\b|ness\b|ist\b|ism\b)', word):
                return '_NOUN_'
            elif re.search(r'(ate\b|fy\b|ize\b|\ben|\bem)', word):
                return '_VERB_'
            elif re.search(r'(\bun|\bin|ble\b|ry\b|ish\b|ious\b|ical\b|\bnon)', word):
                return '_ADJ_'
            else:
                return "_RARE_"
        else:
            return word
    else:
        if not re.search(r'\w', word):
            return '_PUNCS_'
        elif re.search(r'[A-Z]', word):
            return '_CAP_'
        elif word.isupper():
            return "_ALLCAP_"
        elif re.search(r'\d', word):
            return '_NUM_'
        elif re.search(r'(ses\b|ion\b|ty\b|ics\b|ment\b|ence\b|ance\b|ness\b|ist\b|ism\b)', word):
            return '_NOUN_'
        elif re.search(r'(ate\b|fy\b|ize\b|\ben|\bem)', word):
            return '_VERB_'
        elif re.search(r'(\bun|\bin|ble\b|ry\b|ish\b|ious\b|ical\b|\bnon)', word):
            return '_ADJ_'
        else:
            return "_RARE_"


f = open("gene.train", "r")
# f2 = open("gene.train_unk", "w")
f3 = open("gene.train_group_no_label", "w")
f2 = open("gene.train_group", "w")
# counts = []
lines = f.readlines()
for line in lines:
    if line != "\n":
        temp = line.strip("\n").split(" ")
        word = temp[0]
        temp[0] = group_word(word)
        f2.write(" ".join(temp) + "\n")
        f3.write(temp[0] + "\n")
    else:
        f2.write("\n")
        f3.write("\n")
f3.write("\n")

f.close()
f2.close()
f3.close()

counts_file = "gene.counts_group"
words = {}
ngrams = {1: {}, 2: {}, 3: {}}
word_counts = {}
lines = []
for l in open(counts_file):
    lines.append(l)
    t = l.strip().split()
    count = int(t[0])
    key = tuple(t[2:])
    if t[1] == "1-GRAM":
        ngrams[1][key[0]] = count
    elif t[1] == "2-GRAM":
        ngrams[2][key] = count
    elif t[1] == "3-GRAM":
        ngrams[3][key] = count
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
                O = emission[(w, "O")]
                I_GENE = emission[(w, "I-GENE")]
            except:
                if O == None:
                    O = 0
                if I_GENE == None:
                    I_GENE = 0
        else:
            w = group_word(w, exist=False)
            O = emission[(w, "O")]
            I_GENE = emission[(w, "I-GENE")]
        if O > I_GENE:
            tags.append("O")
        else:
            tags.append("I-GENE")
    return tags


try:
    f = open("gene_dev_group.p1.out", "x")
except:
    f = open("gene_dev_group.p1.out", "w")

f2 = open("gene.dev", "r")
lines = f2.readlines()
cur = []
count = 0
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

try:
    f = open("gene_train_group.p2.out", "x")
except:
    f = open("gene_train_group.p2.out", "w")

f2 = open("gene.train_group_no_label", "r")
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