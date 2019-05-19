import numpy as np
import dynet as dy
import json
import random
import nltk

INPUT_DIM=1500
HIDDEN_DIM=200

lines = []
with open("unim_poem.json") as json_file:
    data = json.load(json_file)
    for p in data:
        [lines.append('s ' + line + ' /s') for line in p['poem'].split("\n")]

bigram_dict = {}
wordToIndex = {}
indexToWord = {}

index2 = 0
corpus_list = []
for line in lines:
    line = line.replace("youve", "you have")
    line = line.replace("'ry", "ery")
    w_list = nltk.word_tokenize(line)
    w_list = [word for word in w_list if word.isalpha()]
    for index in range(len(w_list) - 1):
        if index2 < INPUT_DIM:
            index2 += 1
        else:
            break

        [corpus_list.append(w_list[index])]
        try:
            bigram_dict[w_list[index]][w_list[index + 1]] += 1
        except:
            try:
                bigram_dict[w_list[index]][w_list[index + 1]] = 1
            except:
                bigram_dict[w_list[index]] = {w_list[index + 1]: 1}

    if index2 >= INPUT_DIM:
        break


index = 0
for dict_items in bigram_dict.items():
    indexToWord[index] = dict_items[0]
    wordToIndex[dict_items[0]] = index
    index += 1
indexToWord[index] = "/s"
wordToIndex["/s"] = index



INPUT_DIM = len(wordToIndex)
print(INPUT_DIM)
m = np.zeros(INPUT_DIM)
one_hot_vectors = []
for w in corpus_list:
    index = wordToIndex[w]
    m[index] = 1
    one_hot_vectors.append(m.copy())
    m[index] = 0


x1 = dy.vecInput(INPUT_DIM)
model = dy.Model()
pW = model.add_parameters((HIDDEN_DIM, INPUT_DIM))
pb = model.add_parameters(HIDDEN_DIM)
pD = model.add_parameters(INPUT_DIM)
pU = model.add_parameters((INPUT_DIM, HIDDEN_DIM))

dy.renew_cg()

total_loss = 0
seen_instances = 0
trainer = dy.SimpleSGDTrainer(model)
for epoch in range(25):
    for i in range(len(one_hot_vectors) - 1):
        dy.renew_cg()
        w = dy.parameter(pW)
        b = dy.parameter(pb)
        d = dy.parameter(pD)
        u = dy.parameter(pU)
        x = dy.inputVector(one_hot_vectors[i])
        y = dy.inputVector(one_hot_vectors[i+1])
        output = dy.softmax(u * (dy.tanh((w * x) + b)) + d)


        loss =  -dy.log(dy.dot_product(output, y))
        seen_instances += 1
        total_loss += loss.value()
        loss.forward()
        loss.backward()
        trainer.update()

    print("average loss is:", total_loss / seen_instances)

# ii = wordToIndex["glad"]
# ii2 = wordToIndex["smiles"]
# m[ii] = 1
# x.set(m.copy())
# selam = output.npvalue()
# print(ii2)
# # print(np.where(selam == selam.max()))
# print(indexToWord[np.argmax(selam)])
# print(bigram_dict["glad"])
# m[ii] = 0


def changeStartWord(ind=0):
    if ind == 0:
        ind = random.randint(6, INPUT_DIM)
    index = wordToIndex['s']
    m[index] = 1
    x.set(m.copy())
    m[index] = 0
    sorted_array = np.sort(output.npvalue(), 0)[::-1]
    n_most_similar = np.where(output.npvalue() == sorted_array[ind + 1])

    return indexToWord[n_most_similar[0][0]]


def poemGenerator(generated_word="s", line_number=2):
    poems = []
    for poem_index in range(5):
        poem = []
        for bbb in range(line_number):
            line = []
            for word_number in range(5):
                index = wordToIndex[generated_word]
                m[index] = 1
                x.set(m.copy())
                m[index] = 0
                generated_word = indexToWord[np.argmax(output.npvalue())]
                if generated_word == "s":
                    generated_word = changeStartWord()
                elif generated_word == "/s":
                    break
                line.append(generated_word)
            poem.append(line)
        poems.append(poem)
        generated_word = changeStartWord(poem_index)
    return poems

ss = poemGenerator()
for line in ss:
    print(line)
