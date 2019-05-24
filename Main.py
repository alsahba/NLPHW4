import numpy as np
import dynet as dy
import json
import random
import nltk
import math
import bisect
from nltk.corpus import stopwords


CORPUS_LIMIT = 100
bigram_dict, wordToIndex, indexToWord = {}, {}, {}
one_hot_vectors = []


def readPoems():
    sentences = []
    stop_words = set(stopwords.words('english'))
    with open("unim_poem.json") as json_file:
        data = json.load(json_file)
        [sentences.append('SOTL ' + line + ' EOTL') for poem in data for line in poem['poem'].split("\n")]
    return sentences


def replacer(sentence):
    sentence = sentence.replace("youve", "you have")
    sentence = sentence.replace("'ry", "ery")
    sentence = sentence.replace("th'", "the")
    sentence = sentence.replace("nae ither", "neither")
    sentence = sentence.replace("'n", "en")
    return sentence


def buildCountBasedBigramDictionary(sentences):
    index2 = 0
    corpus_list = []
    for sentence in sentences:
        sentence = replacer(sentence)
        w_list = [word for word in nltk.word_tokenize(sentence) if word.isalpha()]

        for index in range(len(w_list) - 1):
            corpus_list.append(w_list[index])
            if index2 < CORPUS_LIMIT:
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

        corpus_list.append("EOTL")

        if index2 >= CORPUS_LIMIT:
            break
    return corpus_list


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


def changeStartWord():
    ind = random.randint(0, INPUT_DIM - 1)
    start_word = indexToWord[ind]
    while start_word == "SOTL" or start_word == "EOTL":
        ind = random.randint(0, INPUT_DIM)
        start_word = indexToWord[ind]
    return start_word


def generateNewWord(generated_word, generic_zero_vector, min_word_limit):
    x.set(createWordOneHotVector(generated_word, generic_zero_vector))
    new_word = cumulativelyGenerate(output.npvalue())
    if new_word == "EOTL" and min_word_limit > 0:
        while (new_word == "EOTL" or new_word == "SOTL"):
            new_word = cumulativelyGenerate(output.npvalue())
    return new_word


def generatePoemLine(generated_word):
    poem_line = []
    min_word_limit = 5
    for number_of_words in range(150):
        generated_word = generateNewWord(generated_word, generic_zero_vector, min_word_limit)
        if generated_word == "EOTL":
            break
        poem_line.append(generated_word)
        min_word_limit -= 1
    return poem_line


def generatePoems(generated_word="SOTL", line_number=2):
    poems = []
    for number_of_poems in range(5):
        new_poem = []
        [new_poem.append(generatePoemLine(generated_word)) for x in range(line_number)]
        poems.append(new_poem)
        generated_word = changeStartWord()
    return poems


def buildHelperDictionaries():
    index = 0
    for dict_items in bigram_dict.items():
        indexToWord[index] = dict_items[0]
        wordToIndex[dict_items[0]] = index
        index += 1
    indexToWord[index] = "EOTL"
    wordToIndex["EOTL"] = index


def createOneHotVector(index, generic_zero_vector):
    generic_zero_vector[index] = 1
    one_hot_vector = generic_zero_vector.copy()
    generic_zero_vector[index] = 0
    return one_hot_vector


def createWordOneHotVector(word, generic_zero_vector):
    index = wordToIndex[word]
    generic_zero_vector[index] = 1
    one_hot_vector = generic_zero_vector.copy()
    generic_zero_vector[index] = 0
    return one_hot_vector


def cumulativelyGenerate(one_hot_vector):
    cumulative_probability = 0
    breakpoints = []
    words = []
    for index, probability in enumerate(one_hot_vector):
        cumulative_probability += probability
        breakpoints.append(cumulative_probability)
        words.append(indexToWord[index])

    dice = random.uniform(0, 1)
    convert_breakpoint_to_index = bisect.bisect(breakpoints, dice)
    return words[convert_breakpoint_to_index]


sentences = readPoems()
corpus_list = buildCountBasedBigramDictionary(sentences)
buildHelperDictionaries()

INPUT_DIM = len(wordToIndex)
HIDDEN_DIM = 100
print(INPUT_DIM)
generic_zero_vector = np.zeros(INPUT_DIM)

for word in corpus_list:
    try:
        index = wordToIndex[word]
        one_hot_vectors.append(createOneHotVector(index, generic_zero_vector))
    except:
        pass


model = dy.Model()
pW = model.add_parameters((HIDDEN_DIM, INPUT_DIM))
pb = model.add_parameters(HIDDEN_DIM)
pD = model.add_parameters(INPUT_DIM)
pU = model.add_parameters((INPUT_DIM, HIDDEN_DIM))

total_loss = 0
seen_instances = 0
trainer = dy.SimpleSGDTrainer(model)
for epoch in range(10):
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

ss = generatePoems()
for line in ss:
    print(line)
c = calculatePerplexity(ss, 2)
print(c)