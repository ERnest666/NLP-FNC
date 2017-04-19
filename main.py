import csv, sys, nltk
import gensim
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from scipy import spatial
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.dataset import DataSet
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import sentence2vec as s2v
import re

# Replace words using regular expression
replacement_patterns = [
   (r'won\'t', 'will not'),
   (r'can\'t', 'cannot'),
   (r'i\'m', 'i am'),
   (r'ain\'t', 'is not'),
   (r'(\w+)\'ll', '\g<1> will'),
   (r'(\w+)n\'t', '\g<1> not'),
   (r'(\w+)\'ve', '\g<1> have'),
   (r'(\w+)\'s', '\g<1> is'),
   (r'(\w+)\'re', '\g<1> are'),
   (r'(\w+)\'d', '\g<1> would')
]
class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    # return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))
    return nltk.word_tokenize(text.translate(remove_punctuation_map))

### Performance becomes better after removing the normalize part
# vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
vectorizer = TfidfVectorizer(stop_words='english', strip_accents = 'unicode', sublinear_tf = True, smooth_idf = True)   ## return term-document matrix

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

bodies = sys.argv[1]
train = sys.argv[2]
dev = sys.argv[3]
# outputfile = sys.argv[4]

train_dataset = DataSet(bodies, train)
test_dataset = DataSet(bodies, dev)

train_bodies = train_dataset.articles
train_headlines = train_dataset.stances

test_bodies = test_dataset.articles
test_headlines = test_dataset.stances

                            ## For backup
train_document_arr = []     # full set of original headline and body pair
train_stances_arr = []      # full set of original training stances
                            ## two phase of training
train_relunrel_arr = []     # Store unrelated and related(agree, disagree, discuss) stances
train_rel_arr = []          # Store stances within related
train_rel_sim_values = []   # Cosine similarity values of only agree, disagree, discuss stances
train_sim_values = []       # Cosine similarity values of all stances
                            ## For testing
test_rel_sim_values = []    # Cosine similarity values of only agree, disagree, discuss stances
test_sim_values = []        # Cosine similarity values of all stances


## get headline and body pair and stances(only related and unrelated) from training data
print('Loading training data and computing similarity...')
for i in range(len(train_headlines)):
    ##### For training data
    pair = []
    sim_value = []
    pair.append(train_headlines[i]['Headline'])
    pair.append(train_bodies[train_headlines[i]['Body ID']])

    # train_document_arr.append(train_headlines[i]['Headline'].translate(trantab)+train_bodies[train_headlines[i]['Body ID']].translate(trantab))
    train_document_arr.append(pair)
    train_stances_arr.append(train_headlines[i]['Stance'])
    sim_value.append(cosine_sim(train_headlines[i]['Headline'], train_bodies[train_headlines[i]['Body ID']]))

    if train_headlines[i]['Stance'] == 'unrelated':
        train_relunrel_arr.append('unrelated')
    else:
        train_relunrel_arr.append('related')
        # train_rel_arr.append(train_headlines[i]['Stance'])
        train_rel_sim_values.append(sim_value)
        if train_headlines[i]['Stance'] == 'agree':
            train_rel_arr.append([0,0])
        elif train_headlines[i]['Stance'] == 'discuss':
            train_rel_arr.append([0,1])
        elif train_headlines[i]['Stance'] == 'disagree':
            train_rel_arr.append([1,0])

    # print(train_headlines[i]['Stance'], cosine_sim(train_headlines[i]['Headline'], train_bodies[train_headlines[i]['Body ID']]))
    train_sim_values.append(sim_value)


print('Loading test data and computing similarity...')
for i in range(len(test_headlines)):
    ##### For test data
    pair = []
    sim_value = []
    sim_value.append(cosine_sim(test_headlines[i]['Headline'], train_bodies[test_headlines[i]['Body ID']]))
    test_sim_values.append(sim_value)

#######################################################
#### OneVsRestClassifier(LinearSVC(random_state=0))####
#######################################################
## train classifier with similarity values and training labels with train_sim_values and train_stances_arr
# print(OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X))
### Phase 1
print('Training phase 1 classifier...')
phase1_X_train = train_sim_values
phase1_y_train = train_relunrel_arr
phase1_classifier = OneVsRestClassifier(SVC(random_state=0, C=1.4)).fit(phase1_X_train, phase1_y_train)
# ### Phase 2
# phase2_X_train = np.asarray(train_rel_sim_values)
# phase2_y_train = np.asarray(train_rel_arr)
# phase2_classifier = OutputCodeClassifier(SVC(random_state=0),code_size=2, random_state=0)
# phase2_classifier = phase2_classifier.fit(phase2_X_train, phase2_y_train)
### Alt: random forest
# forest = RandomForestClassifier(n_estimators=100, random_state=1)
# multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
# multi_target_forest.fit(phase2_X_train, phase2_y_train)

### Phase 1
phase1_X_test = test_sim_values
print('Phase 1 Predicting...')
phase1_predict = phase1_classifier.predict(phase1_X_test)
for idx, pre in enumerate(phase1_predict):
    if phase1_predict[idx] == 'unrelated':
        test_headlines[idx]['Stance'] = phase1_predict[idx]
    else:
        test_headlines[idx]['Stance'] = 'related'   # for now
        ## get test_rel_sim_values
        test_rel_sim_values.append(test_sim_values[idx])

### Phase 2
print('Phase2 begins...')
headline = []
bodyID = []
stances = []
for row in train_headlines:
    if row['Stance'] == 'unrelated':
        continue
    headline.append(row['Headline'].lower())
    bodyID.append(int(row['Body ID']))
    stances.append(row['Stance'])

# Read dev.csv
whole_dev_headline = []
dev_bodyID = []
dev_stances = []
for row in test_headlines:
    whole_dev_headline.append(row['Headline'].lower())
    dev_bodyID.append(int(row['Body ID']))
    dev_stances.append(row['Stance'])
result = []
result.append(whole_dev_headline)
result.append(dev_bodyID)
result.append(dev_stances)

dev_headline = []
for index in range(len(whole_dev_headline)):
    if dev_stances[index] == 'related':
        dev_headline.append(whole_dev_headline[index])

# with open('bodies.csv', 'r', encoding='utf-8', errors='ignore') as f:
#     reader = csv.reader(f)
#     l = list(reader)
# l = l[1 : len(l)]
body_dic = {}
for key in train_bodies:
    body_dic[key] = sent_tokenize(train_bodies[key].lower())

# Replace words and tokenize
replacer = RegexpReplacer()
tokenizer = TreebankWordTokenizer()
for i in range(len(headline)):
    headline[i] = tokenizer.tokenize(replacer.replace(headline[i]))
for i in range(len(whole_dev_headline)):
    whole_dev_headline[i] = tokenizer.tokenize(replacer.replace(whole_dev_headline[i]))
for key in body_dic:
    new_body = []
    for sentence in body_dic[key]:
        new_body.append(tokenizer.tokenize(replacer.replace(sentence)))
    body_dic[key] = new_body

corpus = []
stops = set(stopwords.words('english'))
# Remove punctuations and stopwords
x = re.compile('[%s]' % re.escape(string.punctuation))
new_headline = []
for h in headline:
    new_h = []
    for token in h:
        new_token = x.sub(u'', token)
        if not new_token == u'' and new_token not in stops:
            new_h.append(new_token)
    if len(new_h) >= 1:
        corpus.append(new_h)
        new_headline.append(new_h)
new_dev_headline = []
for h in whole_dev_headline:
    new_h = []
    for token in h:
        new_token = x.sub(u'', token)
        if not new_token == u'' and new_token not in stops:
            new_h.append(new_token)
    if len(new_h) >= 1:
        corpus.append(new_h)
        new_dev_headline.append(new_h)
for key in body_dic:
    new_sen = []
    for sentence in body_dic[key]:
        new_b = []
        for token in sentence:
            new_token = x.sub(u'', token)
            if not new_token == u'' and new_token not in stops:
                new_b.append(new_token)
        if len(new_b) >= 1:
            corpus.append(new_b)
            new_sen.append(new_b)
    body_dic[key] = new_sen

headline = new_headline
whole_dev_headline = new_dev_headline

# Word2Vec
print('Phase 2 word2vec...')
model = gensim.models.Word2Vec(corpus, min_count=1, size=330, alpha=0.125)
model.save('word_vec')
# model = gensim.models.Word2Vec.load('word_vec')
#
# Sentence2Vec
print('Phase 2 sentence2vec...')
embedding_size = 330
count = 0
headline_vectors = {}
body_vectors = {}
for h in headline:
   words = []
   for token in h:
       words.append(s2v.Word(token, model.wv[token]))
   sentence = s2v.Sentence(words)
   headline_vectors[count] = s2v.sentence_to_vec([sentence], embedding_size)[0]
   count = count + 1
count = 0
for key in body_dic:
    num_of_sentence = 0
    sentence_vectors = []
    for sentence in body_dic[key]:
        words = []
        for token in sentence:
            words.append(s2v.Word(token, model.wv[token]))
        new_sentence = s2v.Sentence(words)
        current_vetor = s2v.sentence_to_vec([new_sentence], embedding_size)

        if num_of_sentence == 0:
            sentence_vectors = current_vetor
        else:
            for i in range(embedding_size):
                sentence_vectors[0][i] = sentence_vectors[0][i] + current_vetor[0][i]
        num_of_sentence = num_of_sentence + 1
    for i in range(embedding_size):
        sentence_vectors[0][i] = sentence_vectors[0][i]/num_of_sentence
    body_vectors[key] = sentence_vectors[0]

# print(len(headline_vectors))
# Save the sentence vectors
# def save_vec(file, vector):
#     with open(file, 'w') as f:
#         for i in range(len(vector)):
#             for j in range(embedding_size):
#                 f.write(str(vector[i][j]) + ' ')
#             f.write('\n')
#
# save_vec('headline_vec.txt', headline_vectors)
# save_vec('body_text_vec.txt', body_vectors)
#
# def load(file, dictionary):
#     count = 0
#     if file == 'body_text_vec.txt':
#         with open('bodies.csv', 'r', encoding='utf-8', errors='ignore') as f:
#             reader = csv.reader(f)
#             l = [row[0] for row in list(reader)]
#             l = l[1 : len(l)]
#     with open(file, 'r') as f:
#         for line in f:
#             if file == 'body_text_vec.txt':
#                 index = int(l[count])
#             else:
#                 index = count
#             dictionary[index] = [float(num) for num in line.split(' ')[0 : 360]]
#             count = count + 1
#
# headline_vec = {}
# body_vec = {}
# load('headline_vec.txt', headline_vec)
# load('body_text_vec.txt', body_vec)

vector = []
def cosine_simi(vec1, vec2):
    cos_sim = 1 - spatial.distance.cosine(vec1, vec2)
    return cos_sim
for i in range(len(headline_vectors)):
    v = cosine_simi(headline_vectors[i], body_vectors[int(bodyID[i])])
    vector.append([v])

dev_headline_vec = []
for i in range(len(whole_dev_headline)):
    if dev_stances[i] == 'related':
        words = []
        for token in whole_dev_headline[i]:
            words.append(s2v.Word(token, model.wv[token]))
        sentence = s2v.Sentence(words)
        dev_headline_vec.append(s2v.sentence_to_vec([sentence], embedding_size)[0])

test_vector = []
for i in range(len(dev_headline_vec)):
    v = cosine_simi(dev_headline_vec[i], body_vectors[int(dev_bodyID[i])])
    test_vector.append([v])
print('Training phase 2 classifier...')
# clf = ensemble.RandomForestClassifier(n_estimators=25)
clf = ensemble.GradientBoostingClassifier(learning_rate=0.14, n_estimators=115, max_depth=5)
clf.fit(vector, stances)
print('Phase 2 predicting and writing out results...')
prediction = clf.predict(test_vector)
count = 0
with open('answer.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Headline', 'Body ID', 'Stance'])
    for i in range(len(result[0])):
        h = result[0][i]
        b = result[1][i]
        s = result[2][i]
        if s == 'related':
            s = prediction[count]
            count = count + 1
        csvwriter.writerow([h, b, s])