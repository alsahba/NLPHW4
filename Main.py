import numpy as np
import dynet as dy
import json
import random
import nltk
import math

INPUT_DIM=1000
HIDDEN_DIM=500

lines = []
with open("unim_poem.json") as json_file:
    data = json.load(json_file)
    for p in data:
        [lines.append('SOTL ' + line + ' EOTL') for line in p['poem'].split("\n")]

bigram_dict, wordToIndex, indexToWord = {}, {}, {}

index2 = 0
corpus_list = []
for line in lines:
    line = line.replace("youve", "you have")
    line = line.replace("'ry", "ery")
    line = line.replace("th'", "the")
    line = line.replace("nae ither", "neither")
    line = line.replace("'n", "en")
    w_list = [word for word in nltk.word_tokenize(line) if word.isalpha()]
    [corpus_list.append(w) for w in w_list]
    for index in range(len(w_list) - 1):
        if index2 < INPUT_DIM:
            index2 += 1
        else:
            break

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
indexToWord[index] = "EOTL"
wordToIndex["EOTL"] = index


def createOneHotVector(index):
    generic_zero_vector[index] = 1
    one_hot_vector = generic_zero_vector.copy()
    generic_zero_vector[index] = 0
    return one_hot_vector


INPUT_DIM = len(wordToIndex)
print(INPUT_DIM)
generic_zero_vector = np.zeros(INPUT_DIM)
one_hot_vectors = []
for word in corpus_list:
    try:
        index = wordToIndex[word]
        one_hot_vectors.append(createOneHotVector(index))
    except:
        pass


model = dy.Model()
pW = model.add_parameters((HIDDEN_DIM, INPUT_DIM))
pb = model.add_parameters(HIDDEN_DIM)
pD = model.add_parameters(INPUT_DIM)
pU = model.add_parameters((INPUT_DIM, HIDDEN_DIM))
# m2 = dy.ParameterCollection()
# pW = m2.add_parameters((HIDDEN_DIM, INPUT_DIM))
# pb = m2.add_parameters(HIDDEN_DIM)
# pD = m2.add_parameters(INPUT_DIM)
# pU = m2.add_parameters((INPUT_DIM, HIDDEN_DIM))
# m2.populate("tmp.model")
#
# # w = dy.parameter(pW)
# # b = dy.parameter(pb)
# # d = dy.parameter(pD)
# # u = dy.parameter(pU)
# # x = dy.inputVector(one_hot_vectors[0])
# # y = dy.inputVector(one_hot_vectors[1])
# # output = dy.softmax(u * (dy.tanh((w * x) + b)) + d)



dy.renew_cg()
total_loss = 0
seen_instances = 0
trainer = dy.SimpleSGDTrainer(model)
for epoch in range(20):
    for i in range(len(one_hot_vectors) - 1):
        dy.renew_cg()
        w = dy.parameter(pW)
        b = dy.parameter(pb)
        d = dy.parameter(pD)
        u = dy.parameter(pU)
        x = dy.inputVector(one_hot_vectors[i])
        y = dy.inputVector(one_hot_vectors[i+1])

        output = dy.softmax(u * (dy.tanh((w * x) + b)) + d)
        loss = -dy.log(dy.dot_product(output, y))

        seen_instances += 1
        total_loss += loss.value()
        loss.forward()
        loss.backward()
        trainer.update()

    print("average loss is:", total_loss / seen_instances)


try:
    ii = wordToIndex["offended"]
    ii2 = wordToIndex["a"]
    generic_zero_vector[ii] = 1
    x.set(generic_zero_vector.copy())
    selam = output.npvalue()
    print(ii2)
    # print(np.where(selam == selam.max()))
    print(indexToWord[np.argmax(selam)])
    print(bigram_dict["offended"])
    generic_zero_vector[ii] = 0
except:
    pass


def changeStartWord():
    ind = random.randint(0, INPUT_DIM - 1)
    start_word = indexToWord[ind]
    # sorted_array = np.sort(output.npvalue(), 0)[::-1]
    # n_most_similar = np.where(output.npvalue() == sorted_array[0])
    # start_word = indexToWord[n_most_similar[0][0]]
    while start_word == "SOTL" or start_word == "EOTL":
        ind = random.randint(0, INPUT_DIM)
        start_word = indexToWord[ind]

    return start_word


def generatePoems(generated_word="SOTL", line_number=2):
    poems = []
    for number_of_poems in range(5):
        poem = []
        for number_of_lines in range(line_number):
            poem_line = []
            for number_of_words in range(100):
                index = wordToIndex[generated_word]
                x.set(createOneHotVector(index))
                sorted_array = np.sort(output.npvalue(), 0)[::-1]
                n_most_similar = np.where(output.npvalue() == sorted_array[random.randint(1,5)])
                new_word = indexToWord[n_most_similar[0][0]]
                if new_word == "EOTL":
                    if number_of_words < 5:
                        n_most_similar = np.where(output.npvalue() == sorted_array[random.randint(0, 30)])
                        new_word = indexToWord[n_most_similar[0][0]]
                        if new_word == "EOTL":
                            break
                    else:
                        break
                poem_line.append(new_word)
                generated_word = new_word
            poem.append(poem_line)
        poems.append(poem)
        generated_word = changeStartWord()
    return poems

ss = generatePoems()
for line in ss:
    print(line)


def calculateTotalItems(mapping):
    return sum(count for word, count in mapping.items())


def calculatePerplexity(poems, number_of_lines):
    perplexity_array = np.zeros((5,number_of_lines))
    for p_index, poem in enumerate(poems):
        perplexity = 0
        for l_index, poem_line in enumerate(poem):
            for w_index, poem_word in enumerate(poem_line):
                payda = calculateTotalItems(bigram_dict[poem_word])
                try:
                    #todo add one smoothing ekleme
                    pay = bigram_dict[poem_word][poem_line[w_index + 1]]
                    temp_perp = math.log2(pay /payda)
                    perplexity += temp_perp
                except:
                    temp_perp = math.log2(1 / payda)
                    perplexity += temp_perp
            perplexity = (-1 / len(poem_line) ) * perplexity
            perplexity_array[p_index][l_index] = perplexity
            perplexity = 0
    return perplexity_array

c = calculatePerplexity(ss, 2)
print(c)