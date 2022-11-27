import numpy as np
from collections import defaultdict

counts_file = "gene.train_counts_unk"
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


def get_emission(w, tag):
    emission = _emission_baseline()
    O = None
    I_GENE = None
    if w in word_counts:
        try:
            prob = emission[(w, tag)]
            return prob
        except:
            return 0
    else:
        prob = emission[("_RARE_", tag)]
        return prob


def _transition(gram_counts=ngrams[3]):
    trigrams = gram_counts
    transition = defaultdict()
    for key in trigrams:
        a, b, c = key
        transition[key] = trigrams[key] / ngrams[2][(a, b)]
    return transition

transition = _transition()


def trigram_viterbi(sentence):
    start = "*"
    stop = "STOP"
    tagged = []
    taglist = {"O", "I-GENE"}
    pi = defaultdict(float)
    bp = {}
    pi[(0, start, start)] = 1

    # Define tagsets S(k)
    def S(k):
        if k in (-1, 0):
            return {start}
        else:
            return taglist

    n = len(sentence)
    for k in range(1, n + 1):
        for u in S(k - 1):
            for v in S(k):
                max_score = float("-inf")
                max_tag = None
                for w in S(k - 2):
                    score = pi[(k - 1, w, u)] * transition[(w, u, v)] * get_emission(sentence[k - 1], v)
                    if score > max_score:
                        max_score = score
                        max_tag = w
                pi[(k, u, v)] = max_score
                bp[(k, u, v)] = max_tag

    max_score = float('-Inf')
    cur_u_max, cur_v_max = None, None
    for u in S(n - 1):
        for v in S(n):
            score = pi[(n, u, v)] * \
                    transition[(u, v, stop)]
            if score > max_score:
                max_score = score
                cur_u_max = u
                cur_v_max = v

    tags = []
    tags.append(cur_v_max)
    tags.append(cur_u_max)

    for i, k in enumerate(range(n - 2, 0, -1)):
        tags.append(bp[(k + 2, tags[i + 1], tags[i])])
    tags.reverse()
    return tags


# The section is to generate output on trigram, but it takes quite a bit.
# Thus I commented it out and let you decide when to run it but the outcome has been saved

# try:
#     f = open("gene_dev.p1.out_trigram", "x")
# except:
#     f = open("gene_dev.p1.out_trigram", "w")
# f2 = open("gene.dev","r")
# lines = f2.readlines()
# cur = []
# # counter = 0
# transition = _transition()
# for line in lines:
#     if line == "\n":
#         output = trigram_viterbi(cur)
#         for idx,ele in enumerate(output):
#             f.write(cur[idx] + " " + ele + "\n")
#         f.write("\n")
#         cur = []
# #         counter += 1
# #         print(counter)
#     else:
#         line = line.strip("\n")
#         cur.append(line)
#
# f.close()
# f2.close()


