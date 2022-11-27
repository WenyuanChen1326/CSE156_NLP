#!/bin/python

def read_files(tarfname, ngram):
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

    class Data:
        pass

    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar, trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    # from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    sentiment.count_vect = TfidfVectorizer(ngram_range= ngram)
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
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

    class Data:
        pass

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
        (label, text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    print("\nTraining classifier")
    import classify_2_1
    ngrams = [(1,1), (1,2), (1,3), (2,2),(3,3)]
    reg =[0.001, 0.01, 0.1, 1, 5,10,20,50,100,100]
    cur_max = 0
    fig, axs = plt.subplots(5, 2)
    for i, ele in enumerate(ngrams):
        sentiment = read_files(tarfname,ele)
        train_acc_ls = []
        test_acc_ls = []
        for j in reg:
            cls = classify_2_1.train_classifier(sentiment.trainX, sentiment.trainy, j)
            print("\nEvaluating")
            train_acc = classify_2_1.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
            test_acc = classify_2_1.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
            if test_acc > cur_max:
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
    axs[3, 0].set_title("Bigram only  on Train Data")
    axs[3, 1].set_title("Bigram on Test Data")
    axs[4, 0].set_title("Trigram only  on Train Data")
    axs[4, 1].set_title("Trigram only on Test Data")
    fig.text(0.5, 0.04, 'C', ha='center')
    fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()

    print("\n The best model is with ngram = {0} and C = {1}".format(cur_best_ngram, cur_best_C))
    print("\nTraining classifier ")
    cls = classify_2_1.train_classifier(cur_best_model.trainX, cur_best_model.trainy, cur_best_C)

    print("\nEvaluating")
    classify_2_1.evaluate(cur_best_model.trainX, cur_best_model.trainy, cls, 'train')
    classify_2_1.evaluate(cur_best_model.devX, cur_best_model.devy, cls, 'dev')








