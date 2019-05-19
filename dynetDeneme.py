# import dynet as dy

# # define the parameters
# m = dy.ParameterCollection()
# W = m.add_parameters((8,2))
# V = m.add_parameters((1,8))
# b = m.add_parameters((8))
#
# # renew the computation graph
# dy.renew_cg()
#
# # create the network
# x = dy.vecInput(2) # an input vector of size 2.
# output = dy.logistic(V*(dy.tanh((W*x)+b)))
# # define the loss with respect to an output y.
# y = dy.scalarInput(0) # this will hold the correct answer
# loss = dy.binary_log_loss(output, y)
#
# # create training instances
# def create_xor_instances(num_rounds=2000):
#     questions = []
#     answers = []
#     for round in range(num_rounds):
#         for x1 in 0,1:
#             for x2 in 0,1:
#                 answer = 0 if x1==x2 else 1
#                 questions.append((x1,x2))
#                 answers.append(answer)
#     return questions, answers
#
# questions, answers = create_xor_instances()
#
# # train the network
# trainer = dy.SimpleSGDTrainer(m)
#
# total_loss = 0
# seen_instances = 0
# for question, answer in zip(questions, answers):
#     x.set(question)
#     y.set(answer)
#     seen_instances += 1
#     total_loss += loss.value()
#     loss.backward()
#     trainer.update()
#     if (seen_instances > 1 and seen_instances % 100 == 0):
#         print("average loss is:",total_loss / seen_instances)
import nltk
import string
s = "I'm a little bit john and loves it."
s = s.translate(string.punctuation)
l = nltk.word_tokenize(s)
l = [word.lower() for word in l if word.isalpha()]
print(l)
