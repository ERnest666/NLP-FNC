import csv
import gensim as g
count = 0
with open('small_test.csv', 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    l = list(reader)
l = l[1 : len(l)]
headline = [row[0] for row in l]
bodyID = [row[1] for row in l]
stance = [row[2] for row in l]
print(len(headline))
#print(bodyID[0])
#print(stance[0])

count = 0
docLable = []
for i in range(len(headline)):
    docLable.append('SENT' + str(count))
    count = count + 1

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield g.models.doc2vec.TaggedDocument(words=doc.split(),tags=[self.labels_list[idx]])
it = LabeledLineSentence(headline, docLable)

#docs = g.models.doc2vec.TaggedLineDocument(headline)
model = g.models.Doc2Vec(it, size = 300, window = 2, min_count = 1, workers = 1, iter = 100)
model.save('/tmp/headline.txt')
print(model.docvecs[0])