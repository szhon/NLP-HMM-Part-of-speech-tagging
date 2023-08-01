import os
import sys
import argparse
from collections import defaultdict
import itertools

import numpy as np
from tqdm import tqdm
import random
import copy


def read_from_file(filename):
    with open(filename, "r") as file:
        sentences = file.read()
    sentences = sentences.split("\n")
    pair0 = [elem.strip() for elem in sentences[0].split(":")]
    pos_tags = []
    sentence_list = []
    sentence_list.append(pair0)

    for word_num in range(1, len(sentences)):
        sentence = sentences[word_num]

        if sentence:
            curr_word_pair = [elem.strip() for elem in sentence.split(":")]
            word = curr_word_pair[0]
            pos = curr_word_pair[1]
            sentence_list.append(curr_word_pair)
            # the end of a sentence
            if pos == "PUN" and word != ",":
                sentence_list.append(['</s>', '<end>'])
                pos_tags.append(sentence_list)
                sentence_list = []

    print(pos_tags[0])  # output: [['a', 'b'], ['c', 'd']]

    return pos_tags


def read_from_file_test(filename):
    with open(filename, "r") as file:
        sentences = file.read()

    pos_tags = []
    for sentence in sentences.split("\n"):
        if sentence:
            pos_tags.append([elem.strip() for elem in sentence.split(":")])

    print("test file: ")
    print(pos_tags[0])  # output: [['a', 'b'], ['c', 'd']]
    print(pos_tags[1])
    # to void the last empty line
    return pos_tags


class HMMPOS:
    def __init__(self):
        # self.initial_count['DT'] = 5
        self.initial_count = None
        # self.emission_count[('DT', 'an')] =  3
        self.emission_count = None
        self.tag_count = None
        # self.transition_count[('DT','NN')] =  2
        self.transition_count = None
        self.pos_tags = None
        self.observations = None
        self.tag_to_index = None
        self.observation_to_index = None
        self.initial_prob = None
        self.transition_prob = None
        self.emission_prob = None
        self.end_state = '<end>'

    def get_counts(self, train_data, highPos):
        # TODO: store counts to self.initial_count, self.emission_count, self.transition_count
        initial_dic = defaultdict(int)
        emission_dic = defaultdict(int)
        transi_dic = defaultdict(int)

        all_word_dic = defaultdict(int)
        # replace word less than two times to be UNK
        for num_sentence in range(len(train_data)):
            for word_pair_list in train_data[num_sentence]:
                all_word_dic[word_pair_list[0]] += 1

        # unk_list = []
        # for word, frq in all_word_dic.items():
        #     if frq < 2:

        #         unk_list.append(word)

        # count everything
        # initial dic
        for num_sentence in range(len(train_data)):
            # word = train_data[num_sentence][0][0]
            pos = train_data[num_sentence][0][1]

            initial_dic[pos] += 1

        # emission_dic
        for num_sentence in range(len(train_data)):
            for word_pair_list in train_data[num_sentence]:
                word = word_pair_list[0]
                pos = word_pair_list[1]
                # unk
                # if word in unk_list:
                #     word = 'UNK'
                emi_key = (pos, word)
                emission_dic[emi_key] += 1
        """Add to high pos"""
        emission_dic[(highPos, "UNK")]
        print(highPos)
        # transi_dic
        for num_sentence in range(len(train_data)):
            for pair_index in range(len(train_data[num_sentence]) - 1):
                next_index = pair_index + 1
                pos1 = train_data[num_sentence][pair_index][1]
                pos2 = train_data[num_sentence][next_index][1]
                trans_key = (pos1, pos2)
                transi_dic[trans_key] += 1

        self.initial_count = initial_dic
        self.transition_count = transi_dic
        self.emission_count = emission_dic

    def get_lists(self):
        # TODO: store pos tags and vocabulary to self, store their maps to index
        pos_set = set()
        word_set = set()

        # iterate through inital
        for word, frq in self.initial_count.items():
            pos_set.add(word)

        # iterate through emi
        for key, frq in self.emission_count.items():
            pos = key[0]
            word = key[1]
            pos_set.add(pos)
            word_set.add(word)

        # iterate through trans
        for key, frq in self.transition_count.items():
            pos1 = key[0]
            pos2 = key[1]
            pos_set.add(pos1)
            pos_set.add(pos2)

        # to list
        # alphabetic sort
        pos_list = list(pos_set)
        pos_list.sort()
        ob_list = list(word_set)
        ob_list.sort()

        # calculate to index
        pos_dic = defaultdict(int)
        for i in range(len(pos_list)):
            pos_dic[pos_list[i]] = i

        word_dic = defaultdict(int)
        for i in range(len(ob_list)):
            word_dic[ob_list[i]] = i

        self.pos_tags = pos_list
        self.observations = ob_list
        self.tag_to_index = pos_dic
        self.observation_to_index = word_dic

    def get_probabilities(self, initial_k, transition_k, emission_k):
        # TODO: store probabilities in self.initial_prob, self.transition_prob, self.emission_prob

        # initial prob
        T = len(self.pos_tags)
        V = len(self.observations)

        total_initial_count = sum(self.initial_count.values())
        init_prob = np.zeros((len(self.pos_tags)))

        for i in range(len(self.pos_tags)):
            pos = self.pos_tags[i]
            frq = 0
            if pos in self.initial_count.keys():
                frq = self.initial_count[pos]
            prob = (frq + initial_k) / (total_initial_count + T * initial_k)
            init_prob[i] = prob

        self.initial_prob = init_prob

        # trans prob
        # total_trans_count = sum(self.transition_count.values())
        total_trans_count = []
        for i in range(T):
            count = 0
            for j in range(T):
                pos1 = self.pos_tags[i]
                pos2 = self.pos_tags[j]
                key_trans = (pos1, pos2)
                frq = 0
                if key_trans in self.transition_count.keys():
                    frq = self.transition_count[key_trans]
                count += frq
            total_trans_count.append(count)

        trans_prob = np.zeros((T, T))
        for i in range(T):
            for j in range(T):
                pos1 = self.pos_tags[i]
                pos2 = self.pos_tags[j]
                key_trans = (pos1, pos2)
                frq = 0
                if key_trans in self.transition_count.keys():
                    frq = self.transition_count[key_trans]

                prob = (frq + transition_k) / \
                    (total_trans_count[i] + T * transition_k)
                trans_prob[i][j] = prob

        self.transition_prob = trans_prob

        total_emission_count = []
        for i in range(T):
            count = 0
            for j in range(V):
                pos = self.pos_tags[i]
                word = self.observations[j]
                key_emi = (pos, word)
                frq = 0
                if key_emi in self.emission_count.keys():
                    frq = self.emission_count[key_emi]
                count += frq
            total_emission_count.append(count)

        # emission prob

        emi_prob = np.zeros((T, V))
        for i in range(T):
            for j in range(V):
                pos = self.pos_tags[i]
                word = self.observations[j]
                key_emi = (pos, word)
                frq = 0
                if key_emi in self.emission_count.keys():
                    frq = self.emission_count[key_emi]
                prob = (frq + emission_k) / \
                    (total_emission_count[i] + V * emission_k)
                emi_prob[i][j] = prob

        self.emission_prob = emi_prob

    def predict_pos(self, observations):
        pos_tags = []
        # TODO: predict pos tags, you can assume observations are already tokenized
        ob_len = len(observations)
        T = len(self.pos_tags)
        V = len(self.observations)
        pi = np.zeros((ob_len, T))
        previous_tag = np.zeros((ob_len, T))
        # original_obser = observations
        # replace with no UNK
        for i in range(len(observations)):
            word = observations[i]
            if word not in self.observations:
                observations[i] = 'UNK'

        # initialization
        word = observations[0]
        word_index = self.observation_to_index[word]
        for index in range(T):
            inital_prob = self.initial_prob[index]
            e_xs = self.emission_prob[index, word_index]
            pi[0, index] = np.log(inital_prob) + np.log(e_xs)

        # try one for loop
        for j in range(1, ob_len):
            temp = pi[j - 1] + np.log(self.transition_prob).T
            word = observations[j]
            word_index = self.observation_to_index[word]
            pi[j, :] = np.max(temp, axis=1) + \
                np.log(self.emission_prob[:, word_index])
            previous_tag[j] = np.argmax(temp, axis=1)
        # end of trying one for loop

        row_len = np.shape(pi)[0]
        max_pos_index_last = np.argmax(pi[row_len - 1])
        pos_tags.append(self.pos_tags[max_pos_index_last])
        pre_last = previous_tag[row_len - 1, max_pos_index_last]

        for i in reversed(range(0, row_len - 1)):
            # print(i)
            pos_tags.append(self.pos_tags[int(pre_last)])
            pre_last = previous_tag[i][int(pre_last)]

        pos_tags = pos_tags[::-1]
        return pos_tags

    def predict_pos_all(self, sentences):
        # sentences is a list of sentences (each sentence is a list of tokens)
        results = []
        # TODO: append pos tags for each sentence to results
        for i in sentences:
            results.append(self.predict_pos(i))

        return results

    def search_k(self, dev):
        initial_k, transition_k, emission_k = 0.98, 0.99, 0.003
        # TODO: search best k values

        labels = []
        words = []

        best_acc = 0
        for sentence in range(len(dev)):
            label = []
            word = []
            # for sentence in range(len(dev[chunk])):
            labels.append(dev[sentence][1])
            words.append(dev[sentence][0])
            # labels.append(label)
            # words.append(word)

        self.get_probabilities(initial_k, transition_k, emission_k)
        predictions = self.predict_pos_all(words)
        best_acc = get_accuracy(predictions, labels)
        print(f"Best accuracy: {best_acc}")

        return initial_k, transition_k, emission_k

    def test(self, initial_k, transition_k, emission_k, test):
        accuracy = 0
        # TODO: get accuracy on the test set
        words = test

        self.get_probabilities(initial_k, transition_k, emission_k)
        predictions = self.predict_pos_all(words)

        return predictions
        # accuracy = get_accuracy(predictions, labels)

        # return accuracy

    def generate(self, start_tag):
        observations, pos_tags = [], [start_tag]
        # TODO: generate observations and pos tags
        curr_tag = start_tag
        tag_index = self.tag_to_index[curr_tag]
        end_index = self.tag_to_index['<end>']
        while len(pos_tags) <= 15 and tag_index != end_index:
            word_index = np.argmax(self.emission_prob[tag_index])
            curr_word = self.observations[word_index]

            next_pos_index = np.argmax(self.transition_prob[tag_index])
            tag_index = next_pos_index
            cur_pos = self.pos_tags[tag_index]
            pos_tags.append(cur_pos)
            observations.append(curr_word)
            # print(len(observations))
        pos_tags.pop()
        return observations, pos_tags

    def generate_fix(self, start_tag):
        observations, pos_tags = [], [start_tag]
        # TODO: generate observations and pos tags
        curr_tag = start_tag
        tag_index = self.tag_to_index[curr_tag]
        end_index = self.tag_to_index['<end>']
        while len(pos_tags) <= 15 and tag_index != end_index:
            word_choose = random.choices(
                list(self.observations), weights=list(self.emission_prob[tag_index]))
            curr_word = word_choose[0]

            next_pos = random.choices(
                list(self.pos_tags), weights=self.transition_prob[tag_index])
            next_pos = next_pos[0]
            pos_tags.append(next_pos)
            observations.append(curr_word)
            tag_index = self.tag_to_index[next_pos]

        pos_tags.pop()
        return observations, pos_tags


def get_accuracy(predictions, labels):
    accuracy = 0
    correct = 0
    total = 0
    for index in range(len(predictions)):
        correct += np.sum((np.array(predictions[index])
                          == np.array(labels[index])))
        total += len(predictions[index])
    accuracy = correct / total
    return accuracy


def get_highest_freq_pos(train):
    post_tag_frq = defaultdict(int)
    for sentence in train:
        for pair in sentence:
            if pair[1] not in ['</s>', '<end>', 'PUN', 'PUQ', 'PUL', 'PUR']:
                post_tag_frq[pair[1]] += 1

    max_val_dic = max(post_tag_frq.values())
    max_key = []
    for key, value in post_tag_frq.items():
        if value == max_val_dic:
            max_key.append(key)
    return max_key


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        # required=True,
        help="The training files.",
        default="training2.txt"
    )
    parser.add_argument(
        "--testfile",
        type=str,
        # required=True,
        help="One test file.",
        default="test1.txt"
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        # required=True,
        help="The output file.",
        default="output1.txt"
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]
    print("training files are {}".format(training_list))

    print("test file is {}".format(args.testfile))

    print("output file is {}".format(args.outputfile))

    print("Starting the tagging process.")

    train = read_from_file(args.trainingfiles)
    test = read_from_file_test(args.testfile)
    original_test = copy.deepcopy(test)

    print(f"Number of training sentences: {len(train)}")
    print(f"Number of training tokens: {sum([len(i) for i in train])}")
    print(f"Number of test sentences: {len(test)}")
    print(f"Number of test tokens: {sum([len(i) for i in test])}")

    hmm = HMMPOS()
    highPos = get_highest_freq_pos(train)
    hmm.get_counts(train, highPos[0])

    print(f"Length of initial count: {len(hmm.initial_count)}")
    max_ini_tag = max(hmm.initial_count.keys(),
                      key=lambda x: hmm.initial_count[x])
    print(
        f"Max initial tag: {max_ini_tag}\tMax initial count: {hmm.initial_count[max_ini_tag]}")
    print(f"Length of transition count: {len(hmm.transition_count)}")
    max_tran_pair = max(hmm.transition_count.keys(),
                        key=lambda x: hmm.transition_count[x])
    print(
        f"Max transition pair: {max_tran_pair}\tMax transition count: {hmm.transition_count[max_tran_pair]}")
    print(f"Length of emission count: {len(hmm.emission_count)}")
    max_emission_pair = max(hmm.emission_count.keys(),
                            key=lambda x: hmm.emission_count[x])
    print(
        f"Max emission pair: {max_emission_pair}\tMax emission count: {hmm.emission_count[max_emission_pair]}")

    hmm.get_lists()
    print(f"Number of tags: {len(hmm.pos_tags)}")
    print(f"Last 10 POS tags: {hmm.pos_tags[-10:]}")
    print(f"Index of DT: {hmm.tag_to_index['DT']}")
    print(f"Number of observations: {len(hmm.observations)}")
    print(f"Last 10 observations: {hmm.observations[-10:]}")
    print(f"Index of UNK: {hmm.observation_to_index['UNK']}")
    initial_k, transition_k, emission_k = 1, 1, 0.5

    hmm.get_probabilities(initial_k, transition_k, emission_k)
    sample_sentences = ["This is a sentence for testing .".split(),
                        "This is a much more complicated sentence !".split()]
    results = hmm.predict_pos_all(sample_sentences)
    print(results)

    initial_k, transition_k, emission_k = hmm.search_k(train)
    print(initial_k, transition_k, emission_k)

    predictions = hmm.test(initial_k, transition_k, emission_k, test)

    with open(args.outputfile, "w") as file:
        for pair in zip(original_test, predictions):
            file.write(" : ".join([elem[0] for elem in pair]) + "\n")

    # write into output file
