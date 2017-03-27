import csv
import gensim
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
import sentence2vec as s
import re
count = 0

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

with open('train.csv', 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    l = list(reader)
l = l[1 : len(l)]
headline = [row[0].lower() for row in l]
bodyID = [row[1] for row in l]
stance = [row[2] for row in l]

with open('bodies.csv', 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    l = list(reader)
l = l[1 : len(l)]
body_dic = {}
for row in l:
    body_dic[int(row[0])] = sent_tokenize(row[1].lower())

# Replace words and tokenize
replacer = RegexpReplacer()
tokenizer = TreebankWordTokenizer()
for i in range(len(headline)):
    headline[i] = tokenizer.tokenize(replacer.replace(headline[i]))
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
for h in headline:
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

headline = new_headline
# print(corpus[0])
# print(bodyID[0])
# print(stance[0])

# Word2Vec
# model = gensim.models.Word2Vec(corpus, min_count = 1)
# model.save('word_vec')
model = gensim.models.Word2Vec.load('word_vec')

# Sentence2Vec
embedding_size = 100
headline_vectors = []
body_vectors = []
for h in headline:
   words = []
   for token in h:
       words.append(s.Word(token, model.wv[token]))
   sentence = s.Sentence(words)
   headline_vectors.append(s.sentence_to_vec([sentence], embedding_size)[0])
for key in body_dic:
    num_of_sentence = 0
    sentence_vectors = []
    for sentence in body_dic[key]:
        words = []
        for token in sentence:
            words.append(s.Word(token, model.wv[token]))
        new_sentence = s.Sentence(words)
        current_vetor = s.sentence_to_vec([new_sentence], embedding_size)

        if num_of_sentence == 0:
            sentence_vectors = current_vetor
        else:
            for i in range(embedding_size):
                sentence_vectors[0][i] = sentence_vectors[0][i] + current_vetor[0][i]
        num_of_sentence = num_of_sentence + 1
    for i in range(embedding_size):
        sentence_vectors[0][i] = sentence_vectors[0][i]/num_of_sentence
    body_vectors.append(sentence_vectors[0])

# Save the sentence vectors
def save_vec(file, vector):
    with open(file, 'w') as f:
        for i in range(len(vector)):
            for j in range(embedding_size):
                f.write(str(vector[i][j]) + ' ')
            f.write('\n')

save_vec('headline_vec.txt', headline_vectors)
save_vec('body_text_vec.txt', body_vectors)