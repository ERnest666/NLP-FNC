import csv
import gensim
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
import sentence2vec as s2v
import re
from word_2_vector import RegexpReplacer
from sklearn.neural_network import MLPClassifier
headline_vec = {}
body_vec = {}

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

def load(file, dictionary):
    count = 0
    if file == 'body_text_vec.txt':
        with open('bodies.csv', 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            l = [row[0] for row in list(reader)]
            l = l[1 : len(l)]
    with open(file, 'r') as f:
        for line in f:
            if file == 'body_text_vec.txt':
                index = int(l[count])
            else:
                index = count
            dictionary[index] = [float(num) for num in line.split(' ')[0 : 100]]
            count = count + 1

def form(vec1, vec2):
    return vec1 + vec2

load('headline_vec.txt', headline_vec)
load('body_text_vec.txt', body_vec)

with open('train.csv', 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    l = list(reader)
    f.close()
l = l[1 : len(l)]
bodyID = [row[1] for row in l]
stance = [row[2] for row in l]

label = []
for s in stance:
    if s == 'unrelated':
        label.append([0., 0.])
    elif s == 'disagree':
        label.append([0., 1.])
    elif s == 'discuss':
        label.append([1., 0.])
    elif s == 'agree':
        label.append([1., 1.])

vector = []
for i in range(len(headline_vec)):
    v = form(headline_vec[i], body_vec[int(bodyID[i])])
    vector.append(v)

print(len(vector[0]))

clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(100, 100, 50,), random_state=1, max_iter=1000)
clf.fit(vector, label)

with open('dev.csv', 'r', errors='ignore') as f:
    reader = csv.reader(f)
    l = list(reader)
    f.close()
l = l[1 : len(l)]
test_headline = [row[0].lower() for row in l]
test_bodyID = [int(row[1]) for row in l]

with open('bodies.csv', 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    l = list(reader)
    f.close()
l = l[1 : len(l)]
body_dic = {}
for row in l:
    body_dic[int(row[0])] = sent_tokenize(row[1].lower())

replacer = RegexpReplacer()
tokenizer = TreebankWordTokenizer()
for i in range(len(test_headline)):
    try:
        test_headline[i] = tokenizer.tokenize(replacer.replace(test_headline[i]))
    except:
        print(test_headline[i])
        print(i)
        exit()
for key in body_dic:
    new_body = []
    for sentence in body_dic[key]:
        new_body.append(tokenizer.tokenize(replacer.replace(sentence)))
    body_dic[key] = new_body

corpus = []
stops = set('for a of the and to in'.split())
# Remove punctuations and stopwords
x = re.compile('[%s]' % re.escape(string.punctuation))
new_headline = []
for h in test_headline:
    new_h = []
    for token in h:
        new_token = x.sub(u'', token)
        if not new_token == u'' and new_token not in stops:
            new_h.append(new_token)
    if len(new_h) >= 1:
        corpus.append(new_h)
        new_headline.append(new_h)
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
embedding_size = 100
test_headline = new_headline
model = gensim.models.Word2Vec(corpus, min_count = 1, size=100)

headline_vectors = []
body_vectors = []
for h in test_headline:
   words = []
   for token in h:
       words.append(s2v.Word(token, model.wv[token]))
   sentence = s2v.Sentence(words)
   headline_vectors.append(s2v.sentence_to_vec([sentence], embedding_size)[0].tolist())

vector.clear()
for i in range(len(headline_vectors)):
    # print(len(headline_vectors[i]))
    # print(type(headline_vectors[i]))
    # print(len(body_vec[test_bodyID[i]]))
    # print(type(body_vec[test_bodyID[i]]))
    v = form(headline_vectors[i], body_vec[test_bodyID[i]])
    vector.append(v)

prediction = clf.predict(vector)
with open('prediction.txt', 'w') as f:
    for predict in prediction:
        print(predict)
        print(type(predict))
        if predict[0] == 0.:
            if predict[1] == 0.:
                f.writelines('unrelated\n')
            elif predict[1] == 1.:
                f.writelines('disagree\n')
        elif predict[0] == 1.:
            if predict[1] == 0.:
                f.writelines('discuss\n')
            elif predict[1] == 1.:
                f.writelines('agree\n')