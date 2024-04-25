import numpy as np
import pandas as pd

from model import NaiveBayes, NaiveBayesLaplace, LogisticRegression


def partition(df, frac, seed=1234):
    assert 0 < frac <= 1
    df_0 = df[df['y'] == 0].sample(frac=frac, random_state=seed)
    df_1 = df[df['y'] == 1].sample(frac=frac, random_state=seed)
    df_train = pd.concat([df_0, df_1])
    df_test = df.drop(df_train.index)
    return df_train, df_test


def get_data(df):
    y = df['y'].to_numpy()
    y = y.reshape([y.shape[0], 1])

    X = df.drop('y', axis=1)
    X.insert(0, 'x0', [1] * df.shape[0])
    X = X.to_numpy()
    return X, y


def f_score(y_true, y_pred, beta=1.0):
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i] == 1:
            tp += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            fn += 1
        else:
            tn += 1

    if tp == 0:
        return float('nan')
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)


def run_naive_bayes(df, seed, frac):
    df, _ = partition(df, frac, seed=seed)
    df_train, df_test = partition(df, 2 / 3)

    train_X, train_y = get_data(df_train)
    test_X, test_y = get_data(df_test)

    nb_model = NaiveBayes(train_X, train_y, 10, 2)
    nb_model.train()
    pred_y = nb_model.predict(test_X)

    res = 0
    for i in range(len(pred_y)):
        if pred_y[i] == test_y[i]:
            res += 1

    succ = res
    all = len(test_y)
    fscore = f_score(test_y, pred_y, 0.1)

    with open("results/naive_bayes.txt", "a") as f:
        f.write(f'{frac}\t{succ}\t{all}\t{fscore}\n')
        print(f'frac: {frac}\tacc: {succ}/{all}\tf-score: {fscore}')


def run_laplace(df, seed, frac):
    df, _ = partition(df, frac, seed=seed)
    df_train, df_test = partition(df, 2 / 3)

    train_X, train_y = get_data(df_train)
    test_X, test_y = get_data(df_test)

    nb_model = NaiveBayesLaplace(train_X, train_y, 10, 2)
    nb_model.train()
    pred_y = nb_model.predict(test_X)

    res = 0
    for i in range(len(pred_y)):
        if pred_y[i] == test_y[i]:
            res += 1

    succ = res
    all = len(test_y)
    fscore = f_score(test_y, pred_y, 0.1)

    with open("results/laplace.txt", "a") as f:
        f.write(f'{frac}\t{succ}\t{all}\t{fscore}\n')
        print(f'frac: {frac}\tacc: {succ}/{all}\tf-score: {fscore}')


def run_logistic(df, seed, frac):
    df, _ = partition(df, frac, seed=seed)
    df_train, df_test = partition(df, 2 / 3)

    train_X, train_y = get_data(df_train)
    test_X, test_y = get_data(df_test)

    lr_model = LogisticRegression(train_X, train_y)
    thetas, loss = lr_model.train(10000, 0.001)
    pred_y = np.where(lr_model.predict(test_X, thetas[-1]) < 0.5, 0, 1)

    res = 0
    for i in range(len(pred_y)):
        if pred_y[i] == test_y[i]:
            res += 1

    succ = res
    all = len(test_y)
    fscore = f_score(test_y, pred_y, 0.1)

    with open("results/logistic2.txt", "a") as f:
        f.write(f'{frac}\t{succ}\t{all}\t{fscore}\n')
        print(f'frac: {frac}\tacc: {succ}/{all}\tf-score: {fscore}')


if __name__ == '__main__':
    df = pd.read_csv('rp_formatted.data', index_col=False, sep='\t', names=[f'x{i}' for i in range(1, 10)] + ['y'])
    df = df.drop(['y'], axis=1).sub(1).join(df['y'].map({2: 0, 4: 1}))

    with open("results/logistic2.txt", "w") as f:
        pass
    for frac in [0.01, 0.02, 0.125, 0.250, 0.625, 1.0]:
        for seed in range(100):
            run_logistic(df, seed, frac)

#%%
