import os
import sys
import xml.etree.ElementTree as ET
import nltk
import math
import json
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

RELEVANT_FILENAMES = ['cf74.xml', 'cf75.xml',
                      'cf76.xml', 'cf77.xml', 'cf78.xml', 'cf79.xml']
NUMBER_OF_DOCUMENTS = 1239
OUTPUT_FILE_NAME = "ranked_query_docs.txt"
# The thresholds and the parameters of the BM25 were chosen to maximize NDCG@10 score
# and break ties with the F score
RELEVANCE_THRESHOLD_TFIDF = 0.072
RELEVANCE_THRESHOLD_BM = 5
K_PARAM = 1.91
B_PARAM = 0.79
AVGDL = "avgdl"
FILE_LEN = "files lengths"
WORDS_CNT = "words count"
TF = "tf"
IDF = "idf"
VEC_LEN = "vector length"
TFIDF = "tfidf"
BM = "bm25"

# Gets a path to a single file, and for every document(record)
# creates a string that contains all the words in the relevant tags
# and returns a list of tuples of (string_of_words_in_record, record)


def process_file(filepath):
    doc_xml_tree = ET.parse(filepath)
    doc_root = doc_xml_tree.getroot()
    records_list = doc_root.findall("./RECORD")
    tuples_lst = []
    for record in records_list:
        words = ''
        for entry in record:
            tag = entry.tag
            if tag == 'TITLE' or tag == 'ABSTRACT' or tag == 'EXTRACT':
                words += entry.text + ' '
            if tag == 'RECORDNUM':
                record_id = entry.text
        words = words.lower()
        tup = (words, str(int(record_id)))
        tuples_lst.append(tup)
    return tuples_lst

# Fills in a dictionary of words, and for each word keeps its number of occurences
#  in record i (if it's greater than 0)


def update_dict_words(words, record_id, dict_words, dict_max_occur):
    for word in words:
        if not word in dict_words:
            dict_words[word] = {}
        if not record_id in dict_words[word]:
            dict_words[word][record_id] = 0
        dict_words[word][record_id] += 1
    dict_max_occur[record_id] = words.count(max(set(words), key=words.count))

# Computes the tf-score for each word and record(document),
# if word i appears in document j


def compute_tf(dict_tf, dict_max_occur, dict_words):
    for word in dict_words:
        for file in dict_words[word]:
            occurences_of_word_i_in_file_j = dict_words[word][file]
            max_occurences_in_file = dict_max_occur[file]
            tf = occurences_of_word_i_in_file_j / max_occurences_in_file
            if word not in dict_tf:
                dict_tf[word] = {}
            dict_tf[word][file] = tf

# Computes the idf score for each word


def compute_idf(dict_idf, dict_lengths, dict_words):
    D = len(dict_lengths)  # because we saved each file's length
    for word in dict_words:
        df = len(dict_words[word])
        idf = math.log(D/df, 2)
        dict_idf[word] = idf

# Computes vector length value for each file and inserts it to dict_vector_length


def compute_vector_length(dict_vector_length, dict_lengths, dict_tf, dict_idf):
    for file in dict_lengths:
        dict_vector_length[file] = 0
    for word in dict_idf:
        for file in dict_tf[word]:
            dict_vector_length[file] += (dict_tf[word]
                                         [file] * dict_idf[word]) ** 2
    for file in dict_vector_length:
        dict_vector_length[file] = math.sqrt(dict_vector_length[file])

# Takes the path to the directory of the files, and proccesses it by
# calling the function process_file(path).
# Then fills in the dict_words according to each file.
# At the end, it creates the inverted index.


def build_index(path):
    try:
        stop_words = set(stopwords.words("english"))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
    dict_words = {}
    dict_max_occur = {}  # for a file - how many times did the most frequent word occur
    dict_lengths = {}   # for a file - how many words it has
    dict_tf = {}
    dict_idf = {}
    dict_vector_length = {}
    corpus = {}
    for filename in os.listdir(path):
        dir = sys.argv[2]
        if filename in RELEVANT_FILENAMES:
            file_full_path = dir + '/' + filename
            tuples_list = process_file(file_full_path)
            for words, record_id in tuples_list:
                words = tokenizer.tokenize(words)
                words = [ps.stem(word)
                         for word in words if not(word in stop_words)]
                update_dict_words(words, record_id, dict_words, dict_max_occur)
                dict_lengths[record_id] = len(words)
    # Computes the the and idf scores for the inverted index
    compute_tf(dict_tf, dict_max_occur, dict_words)
    compute_idf(dict_idf, dict_lengths, dict_words)
    compute_vector_length(dict_vector_length, dict_lengths, dict_tf, dict_idf)
    D = len(dict_lengths)

    # Computing the average document length, so we can use it for the BM25 afterwards.
    avgdl = sum([dict_lengths[file] for file in dict_lengths])/D
    corpus["avgdl"] = avgdl
    corpus["files lengths"] = dict_lengths
    corpus["words count"] = dict_words
    corpus["tf"] = dict_tf
    corpus["idf"] = dict_idf
    corpus["vector length"] = dict_vector_length
    inverted_index_file = open("vsm_inverted_index.json", "w")
    json.dump(corpus, inverted_index_file, indent=4)
    inverted_index_file.close()

# Gets the path to the index file and returns the inverted index (dictionary)


def get_index(index_path):
    inv_ind_file = open(index_path, 'r')
    inv_ind = json.load(inv_ind_file)
    inv_ind_file.close()
    return inv_ind

# Gets the inverted index and returns the average document length in the corpus
# (document length number of words after stemming and stop words removal)


def get_average_doc_len(inv_ind):
    return inv_ind[AVGDL]

# Gets the inverted index and a number of document and returns
# length of the document from the corpus whose number it is


def get_doc_len(inv_ind, doc_num):
    return inv_ind[FILE_LEN].get(doc_num, 0)

# Gets the inverted index and a number of document and returns
# length of the vector of document from the corpus whose number it is
# (weights vector of TF-IDF)


def get_doc_vec_len(inv_ind, doc_num):
    return inv_ind[VEC_LEN].get(doc_num, 0)


# Gets the inverted index and a word and returns a dictionary
# in which the key is a document number that contain the word
# and the value is the number of the word occurrences in the document.
# If there are no such documents, the function returns None


def get_word_occur_dict(inv_ind, word):
    return inv_ind[WORDS_CNT].get(word, None)


# Gets a dictionary which is returned from the
# function get_word_occur_dict and a document number and returns
# the number of the word (related to the dictionary) occurrences in the document


def get_word_occur_in_doc(word_occur_dict, doc_num):
    return word_occur_dict.get(doc_num, 0)

# Gets the inverted index and a word and returns a dictionary
# in which the key is a document number that contain the word
# and the value is the term frequency of the word in the document.
# If there are no documents in which the word appear, the function returns None


def get_tf_dict(inv_ind, word):
    return inv_ind[TF].get(word, None)

# Gets a dictionary which is returned from the
# function get_tf_dict and a document number and returns
# the term frequency of the word (related to the dictionary) in the document


def get_tf(tf_dict, doc_num):
    return tf_dict.get(doc_num, 0)

# Gets the inverted index and a word and returns the idf score
# of the word in the inverted index (as it computed for the TF-IDF)


def get_idf(inv_ind, word):
    return inv_ind[IDF].get(word, 0)

# Gets the inverted index and a word and returns the number
# of documents from the corpus which contain this word


def compute_num_of_docs_with_word(inv_ind, word):
    word_occur_dict = get_word_occur_dict(inv_ind, word)
    if word_occur_dict:
        return len(word_occur_dict.keys())
    return 0

# Gets the inverted index and a word and returns the idf score
# of the word in the inverted index (as it computed for the BM25)


def compute_idf_of_bm(inv_ind, word):
    n_word = compute_num_of_docs_with_word(inv_ind, word)
    return math.log((((NUMBER_OF_DOCUMENTS - n_word + 0.5) / (n_word + 0.5)) + 1))

# Gets a query (string) and returns the number of occurrences
# of the most common word in the query


def compute_max_occur_of_word_in_query(q):
    query_list = q.split()
    max_occur_word = max(set(query_list), key=query_list.count)
    return query_list.count(max_occur_word)

# Gets a word and a query (string) and returns the term
# frequency of the word in the query


def compute_tf_from_query(word, q):
    return q.count(word) / compute_max_occur_of_word_in_query(q)

# Gets a query vector ( a dictionary with keys of words and values
# of the words' tf-idf scores in the query) and returns it's Euclidean length


def compute_query_vec_len(query_vec):
    length_square = 0
    for tf_idf_score in query_vec.values():
        length_square += tf_idf_score**2
    return math.sqrt(length_square)

# Returns stop words set, tokenizer and port stemmer,
# required for function manipulate_query to operate


def prepare_for_query_manipulation():
    try:
        stop_words = set(stopwords.words("english"))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
    return stop_words, tokenizer, ps

# Gets a query (string) and returns the query after lowering all letters,
# removing all stop word and stemming all the remaining words


def manipulate_query(q):
    stop_words, tokenizer, ps = prepare_for_query_manipulation()
    q = q.lower()
    q = tokenizer.tokenize(q)
    q_list = [ps.stem(word) for word in q if not (word in stop_words)]
    return " ".join(q_list)

# Gets an inverted index, a question (after query_manipulation)
# and an empty dictionary. the functions fills the dictionary such that
# the keys are numbers of relevant documents from the corpus to the question
# and the values are their relevance score based on TF-IDF


def tf_idf_query(inv_ind, question, rank_docs_dict):
    query_vec = dict()
    # keys will be terms and values will be their TF-IDF score in the question
    words_of_question = question.split()
    for word in words_of_question:
        tf_in_question = compute_tf_from_query(word, question)
        idf = get_idf(inv_ind, word)
        query_vec[word] = tf_in_question * idf
        # For word i, this is w_iq from lecture's notation
        word_tf_dict = get_tf_dict(inv_ind, word)
        if word_tf_dict:  # Not empty
            for doc_num in word_tf_dict.keys():
                tf_in_doc = get_tf(word_tf_dict, doc_num)
                if doc_num not in rank_docs_dict:
                    rank_docs_dict[doc_num] = 0  # initialize items for relevant documents
                rank_docs_dict[doc_num] += query_vec[word] * idf * tf_in_doc
                # For word i and document j, add w_iq * w_ij from lecture's notation
    query_vec_len = compute_query_vec_len(query_vec)
    for doc_num in rank_docs_dict:
        doc_vec_len = get_doc_vec_len(inv_ind, doc_num)
        rank_docs_dict[doc_num] = rank_docs_dict[doc_num] / (query_vec_len * doc_vec_len)
        # Normalizes the score

# Gets an inverted index, a question (after query_manipulation)
# and an empty dictionary. the functions fills the dictionary such that
# the keys are numbers of relevant documents from the corpus to the question
# and the values are their relevance score based on BM25


def bm_query(inv_ind, question, rank_docs_dict):
    words_of_question = question.split()
    for word in words_of_question:
        idf = compute_idf_of_bm(inv_ind, word)  # According to the formula from the assignment
        word_occur_dict = get_word_occur_dict(inv_ind, word)
        if word_occur_dict:  # Not empty
            docs_with_word = word_occur_dict.keys()
            for doc_num in docs_with_word:
                if doc_num not in rank_docs_dict:
                    rank_docs_dict[doc_num] = 0  # initialize items for relevant documents
                word_occur_in_doc = get_word_occur_in_doc(word_occur_dict, doc_num)
                doc_len = get_doc_len(inv_ind, doc_num)
                avg_doc_len = get_average_doc_len(inv_ind)
                doc_len_to_avg_len = doc_len / avg_doc_len
                numerator = idf * word_occur_in_doc * (K_PARAM + 1)
                # The numerator in the formula for the current word
                denominator = word_occur_in_doc + K_PARAM * (1 - B_PARAM + B_PARAM * doc_len_to_avg_len)
                # The denominator in the formula for the current word
                rank_docs_dict[doc_num] += (numerator / denominator)

# Gets a ranking (tfidf or bm25) , a path to an inverted index and
# a question and creates a file with the relevant documents in
# descending order of relevance rank. Note that the documents which
# are returned are only those with relevance score which is higher than
# the threshold we chose (according to our model testing).


def query(ranking, index_path, question):
    q = manipulate_query(question)
    inv_ind = get_index(index_path)
    rank_docs_dict = dict()
    # This the dict used for the relevance score of the relevant documents
    if ranking == TFIDF:
        tf_idf_query(inv_ind, q, rank_docs_dict)
        threshold = RELEVANCE_THRESHOLD_TFIDF  # In the beginning of the code
    else:
        bm_query(inv_ind, q, rank_docs_dict)
        threshold = RELEVANCE_THRESHOLD_BM
    sorted_docs_and_scores = sorted(rank_docs_dict.items(), key=lambda x: x[1], reverse=True)
    # Sort in descending order of relevance
    out_file = open(OUTPUT_FILE_NAME, 'w')
    for doc_and_score in sorted_docs_and_scores:
        if doc_and_score[1] > threshold:
            out_file.write(doc_and_score[0])  # Writes the document number
            out_file.write("\n")
        else:
            break
    out_file.close()


if __name__ == "__main__":
    if sys.argv[1] == "create_index":
        build_index(sys.argv[2])
    else:
        query(sys.argv[2], sys.argv[3], sys.argv[4])
