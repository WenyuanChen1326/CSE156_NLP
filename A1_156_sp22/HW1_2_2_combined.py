#!/bin/python

def read_files(tarfname, ngram=(1,1), stopwords = False, stoptokens = 'english', vector_size = 150):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    import string
    import gensim
    import numpy as np
    from gensim.models import Word2Vec
    corpus_train = []
    for w in sentiment.train_data:
        lst_words = w.split()
        corpus_train.append(lst_words)

    punctuation = set(string.punctuation)
    for i in range(len(sentiment.train_data)):
        r = ''.join([c for c in sentiment.train_data[i].lower() if not c in punctuation])
        sentiment.train_data[i] = r
    for i in range(len(sentiment.dev_data)):
        r = ''.join([c for c in sentiment.dev_data[i].lower() if not c in punctuation])
        sentiment.dev_data[i] = r

    # CountVectorizer
    if stopwords:
        sentiment.count_vect = CountVectorizer(stop_words = stoptokens)
        sentiment.trainX_cnt= sentiment.count_vect.fit_transform(sentiment.train_data)
        sentiment.devX_cnt = sentiment.count_vect.transform(sentiment.dev_data)
        # Tfidf
        sentiment.tf_vect = TfidfVectorizer(ngram_range= ngram, stop_words=stoptokens)
        sentiment.trainX_tf = sentiment.tf_vect.fit_transform(sentiment.train_data)
        sentiment.devX_tf = sentiment.tf_vect.transform(sentiment.dev_data)
    else:
        # CountVectorizer
        sentiment.count_vect = CountVectorizer()
        sentiment.trainX_cnt = sentiment.count_vect.fit_transform(sentiment.train_data)
        sentiment.devX_cnt = sentiment.count_vect.transform(sentiment.dev_data)
        # Tfidf
        sentiment.tf_vect = TfidfVectorizer(ngram_range=ngram)
        sentiment.trainX_tf = sentiment.tf_vect.fit_transform(sentiment.train_data)
        sentiment.devX_tf = sentiment.tf_vect.transform(sentiment.dev_data)
        #word2vec
        tokens_train = []
        for w in sentiment.train_data:
            lst_words = w.split()
            tokens_train.append(lst_words)

        model = gensim.models.word2vec.Word2Vec(tokens_train, vector_size=vector_size,
                                                window=6, min_count=1, sg=1)
        vocabs = set(model.wv.index_to_key)
        def get_embedding(tokens):
            valid_words = [w for w in tokens if w in vocabs]
            if valid_words:
                embedding = np.zeros((len(valid_words), vector_size), dtype=float)
                for idx, ele in enumerate(valid_words):
                    embedding[idx] = model.wv[ele]

                return np.mean(embedding, axis=0)
            else:
                return np.zeros(vector_size)

        sentiment.trainX_vec = np.array([get_embedding(words) for words in sentiment.train_data])
        sentiment.devX_vec = np.array([get_embedding(words) for words in sentiment.dev_data])

    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment

def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels
def TfidfVectoeization(stopwords = False):
    import matplotlib.pyplot as plt
    import classify_2_1
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname,stopwords = stopwords)
    if not stopwords: 
        print("\nTraining classifier on cleaned data with all lower clase and punctuation removed")
    else: 
         print("\nTraining classifier on cleaned data with all lower clase ,punctuation and stopwords removed")

    ngrams = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    reg = [0.01, 0.1, 1, 5, 10, 20, 50, 100, 1000]
    cur_max = 0
    cur_max_train = 0
    fig, axs = plt.subplots(5, 2)
    for i, ele in enumerate(ngrams):
        sentiment = read_files(tarfname, ngram = ele, stopwords = stopwords)
        train_acc_ls = []
        test_acc_ls = []
        for j in reg:
            cls = classify_2_1.train_classifier(sentiment.trainX_tf, sentiment.trainy, j)
            print("\nEvaluating")
            train_acc = classify_2_1.evaluate(sentiment.trainX_tf, sentiment.trainy, cls, 'train')
            test_acc = classify_2_1.evaluate(sentiment.devX_tf, sentiment.devy, cls, 'dev')
            if test_acc >= cur_max:
                cur_max_train = train_acc
                cur_max = test_acc
                cur_best_model = sentiment
                cur_best_ngram = ele
                cur_best_C = j
            train_acc_ls.append((train_acc))
            test_acc_ls.append((test_acc))
        axs[i, 0].plot(reg, train_acc_ls)
        axs[i, 1].plot(reg, test_acc_ls)
    axs[0, 0].set_title("Unigram only on Train Data")
    axs[0, 1].set_title("Unigram only on Test Data")
    axs[1, 0].set_title("Uni & Bigram  on Train Data")
    axs[1, 1].set_title("Uni & Bigram on Test Data")
    axs[2, 0].set_title("Uni,Bi & Trigram  on Train Data")
    axs[2, 1].set_title("Uni,Bi & Trigram on Test Data")
    axs[3, 0].set_title("ngrams n = {1-4} on Train Data")
    axs[3, 1].set_title(" ~ on Test Data")
    axs[4, 0].set_title("ngrams n = {1-5} on Train Data")
    axs[4, 1].set_title(" ~ on Test Data")
    fig.text(0.5, 0.04, 'C', ha='center')
    fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()

    print("\n The best tfidf model with less clean is with ngram = {0}, C = {1} and test_accuracy = {2} and train_accuracy = {3}"\
          .format(cur_best_ngram, cur_best_C, cur_max, cur_max_train))

def CountVectorization(stopwords = False):
    import matplotlib.pyplot as plt
    import classify_2_1
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname, stopwords = stopwords)
    if not stopwords: 
        print("\nTraining classifier on cleaned data with all lower clase and punctuation removed")
    else: 
         print("\nTraining classifier on cleaned data with all lower clase ,punctuation and stopwords removed")
    import classify
    cls = classify.train_classifier(sentiment.trainX_cnt, sentiment.trainy)
    print("\nEvaluating")
    classify.evaluate(sentiment.trainX_cnt, sentiment.trainy, cls, 'train')
    classify.evaluate(sentiment.devX_cnt, sentiment.devy, cls, 'dev')
    reg = [0.01, 0.1, 1, 5, 10, 20, 50, 100, 1000]
    train_acc_ls = []
    test_acc_ls = []
    fig1, axs1 = plt.subplots(1, 2)
    cur_max = 0
    cur_max_train = 0
    for i in reg:
        sentiment = read_files(tarfname, stopwords = stopwords)
        cls = classify_2_1.train_classifier(sentiment.trainX_cnt, sentiment.trainy,i)
        print("\nEvaluating")
        a = classify_2_1.evaluate(sentiment.trainX_cnt, sentiment.trainy, cls, 'train')
        b =  classify_2_1.evaluate(sentiment.devX_cnt, sentiment.devy, cls, 'dev')
        if b > cur_max :
            cur_max_train = a
            cur_max = b
            C = i
        train_acc_ls.append(a)
        test_acc_ls.append(b)

    axs1[0].set_title("Train Accuracy from Countvectorizer")
    axs1[1].set_title("Test Accuracy from Countvectorizer")
    axs1[0].plot(reg, train_acc_ls)
    axs1[1].plot(reg, test_acc_ls)
    plt.tight_layout()
    plt.show()
    print("The best train accuracy with CountVectorizer is {}".format(cur_max_train))
    print("The best test accuracy with CountVectorizer is {}".format(cur_max))
    print("the best C para with CountVectorizer is {}". format(C))
def Word2Vec(stopwords = False):
    import matplotlib.pyplot as plt
    import classify_2_1
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    if not stopwords: 
        print("\nTraining classifier on cleaned data with all lower clase and punctuation removed")
    else: 
         print("\nTraining classifier on cleaned data with all lower clase ,punctuation and stopwords removed")
    vector_size = [50,150,300]
    fig1, axs1 = plt.subplots(3, 2)
    cur_max = 0
    cur_max_train = 0
    for inx, ele in enumerate(vector_size):
        sentiment = read_files(tarfname, vector_size = ele, stopwords = stopwords)
        reg = [0.01, 0.1, 1, 5, 10, 20, 50, 100, 1000]
        train_acc_ls = []
        test_acc_ls = []
        for i in reg:
            print("\nTraining classifier on cleaned data with all lower clase and punctuation removed")
            cls = classify_2_1.train_classifier(sentiment.trainX_vec, sentiment.trainy, i)
            print("\nEvaluating")
            a = classify_2_1.evaluate(sentiment.trainX_vec, sentiment.trainy, cls, 'train')
            b = classify_2_1.evaluate(sentiment.devX_vec, sentiment.devy, cls, 'dev')
            train_acc_ls.append(a)
            test_acc_ls.append(b)
            if b > cur_max:
                cur_max_train = a
                cur_max = b
                C = i
                vc = ele
        axs1[inx, 0].plot(reg, train_acc_ls)
        axs1[inx, 1].plot(reg, test_acc_ls)
    axs1[0,0].set_title("Train Accuracy/word2Vec size = 50")
    axs1[0,1].set_title("Test Accuracy/word2Vec size = 50")
    axs1[1, 0].set_title("Train Accuracy/word2Vec size = 150")
    axs1[1, 1].set_title("Test Accuracy/word2Vec size = 150")
    axs1[2, 0].set_title("Train Accuracy/word2Vec size = 300")
    axs1[2, 1].set_title("Test Accuracy /word2Vec size = 300")
    plt.show()
    print("The best param with word2Vec is C = {}, vector size ={} with test accuracy = {} and train accuracy = {}"\
          .format(C, vc, cur_max,cur_max_train))


if __name__ == "__main__":
    CountVectorization(stopwords = True)
    #CountVectorization()
    # Word2Vec()
    # TfidfVectoeization()


    


    






