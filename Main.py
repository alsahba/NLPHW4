import numpy as np
import pandas as pd
import dynet as dy
import json


INPUT_DIM=1000
HIDDEN_DIM=200

lines = []
with open("unim_poem.json") as json_file:
    data = json.load(json_file)
    for p in data:
        [lines.append('<s> ' + line + ' </s>') for line in p['poem'].split("\n")]

bigram_dict = {}
corpusToIndex = {}
indexToCorpus = {}

index2 = 0
liste = []
for line in lines:
    w_list = line.split()
    for index in range(len(w_list) - 1):
        if index2 < INPUT_DIM:
            index2 += 1
        else:
            break

        [liste.append(w_list[index])]
        try:
            bigram_dict[w_list[index]][w_list[index + 1]] += 1
        except:
            try:
                bigram_dict[w_list[index]][w_list[index + 1]] = 1
            except:
                bigram_dict[w_list[index]] = {w_list[index + 1]: 1}

    if index2 >= INPUT_DIM:
        break

index3 = 0
for dict_items in bigram_dict.items():
    indexToCorpus[index3] = dict_items[0]
    corpusToIndex[dict_items[0]] = index3
    index3 += 1

indexToCorpus[index3] = "</s>"
corpusToIndex["</s>"] = index3


m = np.zeros(INPUT_DIM)
k = []
for w in liste:
    index = corpusToIndex[w]
    m[index] = 1
    k.append(m.copy())
    m[index] = 0


x1 = dy.vecInput(INPUT_DIM)
model = dy.Model()
pW = model.add_parameters((HIDDEN_DIM,INPUT_DIM))
pb = model.add_parameters(HIDDEN_DIM)
V = model.add_parameters((INPUT_DIM,HIDDEN_DIM))

W = dy.parameter(pW)
b = dy.parameter(pb)

dy.renew_cg()

x = dy.inputVector(np.ones(INPUT_DIM))
output = dy.logistic(V*(dy.tanh((W*x)+b)))
y = dy.inputVector(np.zeros(INPUT_DIM))
loss = dy.binary_log_loss(output,y)


total_loss = 0
seen_instances = 0
trainer = dy.SimpleSGDTrainer(model)
for epoch in range(25):
    for i in range(len(k) - 1):
        dy.renew_cg()
        x = dy.inputVector(k[i])
        y = dy.inputVector(k[i+1])
        output = dy.softmax((V * (dy.tanh((W * x) + b))))
        loss = dy.binary_log_loss(output, y)
        seen_instances += 1
        total_loss += loss.value()
        loss.forward()
        loss.backward()
        trainer.update()

    print("average loss is:", total_loss / seen_instances)

ii = corpusToIndex["he"]
ii2 = corpusToIndex["waits"]

m[ii] = 1
x.set(m.copy())
selam = output.npvalue()
print(ii2)
print(np.where(selam == selam.max()))
print(indexToCorpus[np.argmax(selam)])
print(bigram_dict["he"])
