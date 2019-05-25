import bisect
import json
import random
import sys
import math
import nltk
import dynet as dy
import numpy as np
from nltk.corpus import stopwords

print("Please enter poem's number of line: ")
number_of_lines = input()
invalid_input_flag = True
while invalid_input_flag:
    try:
       val = int(number_of_lines)
       print("Learning started...")
       invalid_input_flag = False
    except ValueError:
       print("That's not a valid input!")
       print("Please enter poem's number of line: ")
       number_of_lines = input()

# print("If you want put a word limitation on input file enter the amount or press 'q' for running without limitation: ")
# input_limitation = input()
# invalid_input_flag = True
# while invalid_input_flag:
#     try:
#         val = int(input_limitation)
#         print("Yes input string is an Integer.")
#         print("Input number value is: ", val)
#         invalid_input_flag = False
#     except ValueError:
#         if(input_limitation == "q"):
#             input_limitation = 100
#         print("Invalid input!")
#         print("Please enter a limitation or press 'q': ")
#         input_limitation = input()


number_of_lines = int(number_of_lines)
CORPUS_LIMIT = 500
EPOCH = 10
bigram_dict, wordToIndex, indexToWord = {}, {}, {}
one_hot_vectors = []


def readPoems():
    sentences = []
    # stop_words = set(stopwords.words('english'))
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


def calculateTotalItems(unigram_dict):
    total_bigram_count =  sum(count for word, count in unigram_dict.items())
    unique_bigram_count = len(unigram_dict)
    return total_bigram_count + unique_bigram_count


def organizePoemLineForAccuratePerplexity(poem_line):
    p_line = poem_line.copy()
    p_line.insert(len(poem_line), "EOTL")
    p_line.insert(0, "SOTL")
    return p_line


def calculatePerplexity(poems, number_of_lines):
    perplexity_array = np.zeros((5, number_of_lines))
    for p_index, poem in enumerate(poems):
        for l_index, p_line in enumerate(poem):
            poem_line = organizePoemLineForAccuratePerplexity(p_line)
            perplexity = 0
            for w_index in range(len(poem_line) - 1):
                current_word = poem_line[w_index]
                next_word = poem_line[w_index + 1]
                denominator = calculateTotalItems(bigram_dict[current_word])
                try:
                    numerator = bigram_dict[current_word][next_word] + 1
                    perplexity += math.log2(numerator /denominator)
                except:
                    perplexity += math.log2(1 / denominator)

            perplexity = (-1 / len(poem_line)) * perplexity
            perplexity_array[p_index][l_index] = perplexity
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
    new_word = generateWithCumulativeDistribution(output.npvalue())
    if (new_word == "EOTL" or new_word == "SOTL") and min_word_limit > 0:
        while new_word == "EOTL" or new_word == "SOTL":
            new_word = generateWithCumulativeDistribution(output.npvalue())
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


def generatePoems(number_of_lines, generated_word="SOTL"):
    poems = []
    for number_of_poems in range(5):
        new_poem = []
        [new_poem.append(generatePoemLine(generated_word)) for x in range(number_of_lines)]
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


def createWordOneHotVector(word, generic_zero_vector):
    index = wordToIndex[word]
    generic_zero_vector[index] = 1
    one_hot_vector = generic_zero_vector.copy()
    generic_zero_vector[index] = 0
    return one_hot_vector


def generateWithCumulativeDistribution(one_hot_vector):
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


# todo try catch olmayinca kaldirilabilir direkt
for word in corpus_list:
    try:
        one_hot_vectors.append(createWordOneHotVector(word, generic_zero_vector))
    except:
        print("oops")


model = dy.Model()
pW = model.add_parameters((HIDDEN_DIM, INPUT_DIM))
pb = model.add_parameters(HIDDEN_DIM)
pD = model.add_parameters(INPUT_DIM)
pU = model.add_parameters((INPUT_DIM, HIDDEN_DIM))

total_loss = 0
seen_instances = 0

trainer = dy.SimpleSGDTrainer(model)
for epoch in range(EPOCH):
    for index in range(len(one_hot_vectors) - 1):
        dy.renew_cg()
        w = dy.parameter(pW)
        b = dy.parameter(pb)
        d = dy.parameter(pD)
        u = dy.parameter(pU)

        x = dy.inputVector(one_hot_vectors[index])
        y = dy.inputVector(one_hot_vectors[index+1])
        output = dy.softmax(u * (dy.tanh((w * x) + b)) + d)
        
        if y == 0:
            loss = -dy.log(1 - dy.dot_product(output, y))
        else:
            loss = -dy.log(dy.dot_product(output, y))

        seen_instances += 1
        total_loss += loss.value()
        loss.forward()
        loss.backward()
        trainer.update()

    sys.stdout.write("\r%s%s%f%s%d%%" % (
        "Learning is processing...\t", "Average loss: ", total_loss / seen_instances,
        "\t Completion Rate: ", ((epoch / EPOCH) * 100)))
    sys.stdout.flush()
    # print("average loss is:", total_loss / seen_instances)

print("\n\n")
poems = generatePoems(number_of_lines)
perplexity_array = calculatePerplexity(poems, number_of_lines)

def printPoems(poems, perplexity_array):
    for p_index, poem in enumerate(poems):
        print("{} th poem".format(p_index+1))
        for l_index, poem_line in enumerate(poem):
            print("{} \t\t ---> Perplexity of line: {}".format(poem_line, perplexity_array[p_index][l_index]))
            # print(''.join(poem_line).join(("\t\t\t\t --->Perplexity of line: {}").format(perplexity_array[p_index][l_index])))
        print("\n")

printPoems(poems, perplexity_array)