import random
import itertools
import math
from feature_function import FeatureFunction


def load_train_data():
    return load_data('data/train.txt')


def load_test_data():
    return load_data('data/test.txt')


def load_data(file_path):
    '''
    :param file_path: 
    :return:
     [
      (['Shearson', 'is', 'about', '60%-held', 'by', 'American', 'Express', 'Co', '.'], ['NNP', 'VBZ', 'IN', 'JJ', 'IN', 'NNP', 'NNP', 'NNP', '.']),
      (['A.P.', 'Green', 'currently', 'has', '2,664,098', 'shares', 'outstanding', '.'], ['NNP', 'NNP', 'RB', 'VBZ', 'CD', 'NNS', 'JJ', '.']),
      ...
     ]
    '''
    data = []
    with open(file_path) as file:
        sentences = file.read().strip().split('\n\n')
        for sentence in sentences:
            word_list = []
            label_list = []
            for word_info in sentence.split('\n'):
                word, label, _ = word_info.split()
                word_list.append(word)
                label_list.append(label)
            data.append((word_list, label_list))

    return data


def create_feature_functions(train_data):
    features_functions = set()

    for words, labels in train_data:
        for i in range(1, len(labels)):
            features_functions.add(FeatureFunction(labels[i - 1], labels[i]))

    return list(features_functions)


def get_all_labels(train_data):
    available_labels = set()
    for words, labels in train_data:
        available_labels.update(labels)

    return list(available_labels)


def initial_weights(feature_function_size):
    return [random.random() for _ in range(feature_function_size)]


def calc_empirical_expectation(feature_function, train_data):
    empirical_expectation = 0
    for words, labels in train_data:
        for i in range(1, len(labels)):
            empirical_expectation += feature_function.apply(labels[i - 1], labels[i])

    return empirical_expectation


def calc_predicted_expectation(feature_function, train_data, all_labels, feature_functions, weights):
    predicted_expectation = 0
    for words, labels in train_data:
        for predicted_labels in itertools.product(all_labels, repeat=len(words)):
            predicted_expectation += (
                calc_prob_labels_given_words(predicted_labels, words, all_labels, feature_functions, weights) * sum(
                    [feature_function.apply(labels[i - 1], labels[i]) for i in range(1, len(predicted_labels))]))
            print(predicted_expectation)
    return predicted_expectation


def calc_prob_labels_given_words(labels, words, all_labels, feature_functions, weights):
    nominator = 1
    for j in range(1, len(labels)):
        nominator *= math.exp(sum(
            [feature_function.apply(labels[j - 1], labels[j]) * weight for feature_function, weight in
             zip(feature_functions, weights)]))

    denominator = 1
    for predicted_labels in itertools.product(all_labels, repeat=len(words)):
        for j in range(1, len(predicted_labels)):
            nominator *= math.exp(sum(
                [feature_function.apply(predicted_labels[j - 1], predicted_labels[j]) * weight for
                 feature_function, weight in
                 zip(feature_functions, weights)]))
    return nominator / denominator


def train(train_data, all_labels, features_functions):
    learning_rate = 0.1
    iterations = 100
    weights = initial_weights(len(features_functions))

    for _ in range(iterations):
        for i, (feature_function, weight) in enumerate(zip(features_functions, weights)):
            empirical_expectation = calc_empirical_expectation(feature_function, train_data)
            predicted_expectation = calc_predicted_expectation(feature_function, train_data, all_labels,
                                                               features_functions, weights)
            weights[i] = weight + learning_rate * (empirical_expectation - predicted_expectation)


if __name__ == '__main__':
    train_data = load_train_data()
    test_data = load_test_data()

    feature_functions = create_feature_functions(train_data)
    all_labels = get_all_labels(train_data)

    train(train_data, all_labels, feature_functions)
