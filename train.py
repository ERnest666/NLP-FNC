import csv
from sklearn.neural_network import MLPClassifier
headline_vec = {}
body_vec = {}

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

clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(100,), random_state=1, max_iter=1000)
clf.fit(vector, label)

# print(len(headline_vec))
# print(len(body_vec))
# print(len(label))
# print(headline_vec[0])
# print(body_vec[0])

