import bisect
import json
import random
import sys
import math
import nltk
import dynet as dy
import numpy as np
from nltk.corpus import stopwords


# User interaction part, program asks a number for poetry lines of a poem.
# Checks whether the input is valid or not, if valid continues with the user input.
print("Please enter poem's number of line: ")
number_of_lines = input()
invalid_input_flag = True
while invalid_input_flag:
    try:
       number_of_lines = int(number_of_lines)
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


# We defined the epoch number and helper structures.
# Bigram dictionary used for calculations in perplexity.
# Also, we defined two dictionary for detecting word to location and location to word.
CORPUS_LIMIT = 1000
EPOCH = 10
bigram_dict, wordToIndex, indexToWord = {}, {}, {}
one_hot_vectors = []


# This method used for reading json file.
# At the end we have sentences with sentence delimiters.
# SOTL is shortening of Start Of The Line and EOTL is shortening of End Of The Line.
# <s> and </s> tokens is not used because of word tokenizer of nltk deletes /s token after split the </s> token into
# '<', '/s' and '>', this created different problems in different places.
def readPoems():
    sentences = []
    # stop_words = set(stopwords.words('english'))
    with open("unim_poem.json") as json_file:
        data = json.load(json_file)
        [sentences.append('SOTL ' + line + ' EOTL') for poem in data for line in poem['poem'].split("\n")]
    return sentences

# Most of the words in the Json file are misspelled, this method used for correction in the words of a sentence.
def wordCorrecter(sentence):
    sentence = sentence.replace("youve", "you have")
    sentence = sentence.replace("'ry", "ery")
    sentence = sentence.replace("th'", "the")
    sentence = sentence.replace("nae ither", "neither")
    sentence = sentence.replace("'n", "en")
    return sentence


# This method builds a bigram structure with nested dictionaries. We filled the bigram dictionary with bigrams
# and store counts of them. Useful in the perplexity calculation part while calculating conditional probability.
def buildCountBasedBigramDictionary(sentences):
    index2 = 0
    corpus_list = []
    for sentence in sentences:
        sentence = wordCorrecter(sentence)
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


# This method used for determining denominator while doing perplexity calculations.
# Takes unigram dictionary for calculations, this is a dictionary that includes all occurrences with respect to word.
# Firstly calculates total number of occurrences, then calculate unique occurrences and returns summation of them.
# Note: Number of unique bigrams added because of add-one smoothing applied while doing perplexity calculations.
def calculateTotalItems(unigram_dict):
    total_bigram_count =  sum(count for word, count in unigram_dict.items())
    unique_bigram_count = len(unigram_dict)
    return total_bigram_count + unique_bigram_count


# This method used for putting sentence delimiters to generated poetry lines for better perplexity calculations.
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


# This method used for calculating generated poems' perplexities.
# Poem list taken as a parameter, a perplexity array created.
# Then, with nested for loops word by word probability calculated.
# For each poem, perplexity formula applied and result put in the array.
# This perplexity array returned from the method.
def calculatePerplexityPoem(poems):
    perplexity_array = np.zeros(5)
    for p_index, poem in enumerate(poems):
        perplexity, length = 0, 0
        for p_line in poem:
            length += len(p_line)
            poem_line = organizePoemLineForAccuratePerplexity(p_line)
            for w_index in range(len(poem_line) - 1):
                current_word = poem_line[w_index]
                next_word = poem_line[w_index + 1]
                denominator = calculateTotalItems(bigram_dict[current_word])
                try:
                    numerator = bigram_dict[current_word][next_word] + 1
                    perplexity += math.log2(numerator / denominator)
                except:
                    perplexity += math.log2(1 / denominator)
        perplexity = (-1 / length) * perplexity
        perplexity_array[p_index] = perplexity
    return perplexity_array


# This method used for add randomness to the poem generator.
# If a word occurred lots of time there is more likely comes up first word of the poem.
# This method randomly selects a word from the vocabulary of model.
# Also, checks that if the random word is sentence delimiter or not.
def changeStartWord():
    ind = random.randint(0, INPUT_DIM - 1)
    start_word = indexToWord[ind]
    while start_word == "SOTL" or start_word == "EOTL":
        ind = random.randint(0, INPUT_DIM)
        start_word = indexToWord[ind]
    return start_word


# This method is a helper method of generation it creates new word.
# Current word is taken as parameter and next word generated with respect to it.
# Current word's one-hot vector created and given the FNN as input, then output changes with respect to it.
# In the word generation cumulative distribution used, output of the current word's one hot vector give to helper method.
# After the generated word returned from the helper method,
# this method checks whether the new word is sentence delimiter or not.
# If so, helper method called until new word is no more a sentence delimiter.
def generateNewWord(current_word, generic_zero_vector, min_word_limit):
    x.set(createWordOneHotVector(current_word, generic_zero_vector))
    new_word = generateWithCumulativeDistribution(output.npvalue())
    if (new_word == "EOTL" or new_word == "SOTL") and min_word_limit > 0:
        while new_word == "EOTL" or new_word == "SOTL":
            new_word = generateWithCumulativeDistribution(output.npvalue())
    return new_word


# This method is a helper method of poem generation, it creates a new poem line.
# Firstly a list is created, then filled in with newly generated words.
# There were two specific limitations. One of them is minimum word limit for new poem line, it equals to five
# which means that new line's number of words  must be greater or equal than five.
# Other limitation is maximum word limit for a new poem line, it equals to 150
# which means that new line's number of words must be lower or equal than 150.
# Between the two limitations if a sentence end delimiter generated (EOTL), line is considered finished.
# After all that new poem line returned.
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


# This is the head method of poem generation.
# Five poems created in this method and returned as a list.
# Each poems' number of lines taken from the user and created respectively from it.
# NOTE: There is some words occurred a lot in the json file like 'a', 'an', 'the' etc.
# If same word 'SOTL' selected for the first word of the generation,
# It is highly probable that more occurred word will be used as the first word of the poem.
# For this problem changeStartWord method can be used as you wish, if you want to use it, just uncomment it.
# It increases randomness of the generation.
def generatePoems(number_of_lines, generated_word="SOTL"):
    poems = []
    for number_of_poems in range(5):
        new_poem = []
        [new_poem.append(generatePoemLine(generated_word)) for x in range(number_of_lines)]
        poems.append(new_poem)
        # generated_word = changeStartWord()
    return poems


# This method used for building helper dictionaries.
# It iterates all unique words and build a composite double sided dictionary structure.
# NOTE: Because of the bigram's nature EOTL has been never encountered as first word of a bigram.
# So, we add it both dictionaries in the end, because of the iteration of bigram's first words.
def buildHelperDictionaries():
    index = 0
    for dict_items in bigram_dict.items():
        indexToWord[index] = dict_items[0]
        wordToIndex[dict_items[0]] = index
        index += 1
    indexToWord[index] = "EOTL"
    wordToIndex["EOTL"] = index


# This method used for creating one-hot vector of a word.
# It takes word and zero vector as parameters.
# Zero vector is an array filled with zeros and its size equals to FNN model's vocabulary size.
# So, by looking the helper dictionary word's location in the vocabulary detected
# and this location changed to '1' in the zero vector. Then zero vector set to defaults again for the next creations.
# In the end, one-hot vector returned.
def createWordOneHotVector(word, generic_zero_vector):
    index = wordToIndex[word]
    generic_zero_vector[index] = 1
    one_hot_vector = generic_zero_vector.copy()
    generic_zero_vector[index] = 0
    return one_hot_vector


# This method used for generate new word with respect to cumulative distribution of output vector of a word.
# Output vector taken as parameter, two lists are created,
# one of them for words other of them for breakpoints of a distribution.
# Probabilities in the output vector cumulatively added to breakpoints list
# and words added to word list for range matching.
# After iteration is over, random probability selected and in the breakpoints list this probability's index is detected.
# Detected index used in the word list, then we got newly generated word, this word returned.
# NOTE: Bisect library used for range matching between two lists. Random probability is selected for example '0.44',
# in the breakpoints list bisect library find out in what range this probability exists for example '0.35' to '0.46'.
# Range reduced to index and this index used in the word list. I explained with more details in the report.
def generateWithCumulativeDistribution(output_vector):
    cumulative_probability = 0
    breakpoints = []
    words = []
    for index, probability in enumerate(output_vector):
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
# perplexity_array = calculatePerplexity(poems, number_of_lines)
poem_perplexity_array = calculatePerplexityPoem(poems)

def printPoems(poems, perplexity_array):
    for p_index, poem in enumerate(poems):
        print("{}th poem \t -----> \t Perplexity of poem: {}".format(p_index+1, poem_perplexity_array[p_index]))
        for l_index, poem_line in enumerate(poem):
            print(" ".join(poem_line))
            # print("{} \t\t ---> Perplexity of line: {}".format(poem_line, perplexity_array[p_index][l_index]))
            # print(" ".join(poem_line) + " \t\t ---> Perplexity of line: {}".format(perplexity_array[p_index][l_index]))
            # print(''.join(poem_line).join(("\t\t\t\t --->Perplexity of line: {}").format(perplexity_array[p_index][l_index])))
        print("\n")

printPoems(poems, poem_perplexity_array)

