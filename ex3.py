import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def find_bmu(sample, nodes):
    distances = []
    for i_node, node in nodes.iterrows():
        distances.append(np.linalg.norm(sample[:4] - node[:4]))
    return distances.index(min(distances))


def som_fit(dataset, n_epochs=1, verbose=0, init_learning_rate=0.1, finetune_learning_rate=0.1,
            init_neighbor_coef=0.3, finetune_neighbor_coef=0.3, finetune_treshod=50):
    # neuron weights initialization
    nodes = pd.DataFrame(np.random.rand(3, 4))

    # PCA initialization for visualization
    pca = PCA(2)
    scores = pca.fit_transform(dataset.iloc[:, :4])

    # main training loop
    for epoch in range(n_epochs):
        for i_sample, sample in dataset.iterrows():
            # find BMU (index)
            i_bmu = find_bmu(sample, nodes)

            # plot state before step
            if verbose == 2:
                nodes_scores = pca.transform(nodes)

                fig = plt.figure()
                ax = fig.subplots()
                ax.scatter(scores[:, 0], scores[:, 1], c=label_mapper(dataset.iloc[:, 4]), cmap="Set2")
                plt.title("e{:d} - #iter: {:d} - before step".format(epoch, i_sample))
                for i in range(3):
                    if i == i_bmu:
                        color = 'r'
                    else:
                        color = 'black'
                    ax.scatter(nodes_scores[i, 0], nodes_scores[i, 1], c=color)
                ax.scatter(scores[i_sample, 0], scores[i_sample, 1], c='orange', s=40)
                plt.savefig("ex3_outputs/e{:d}_{:d}_before".format(epoch, i_sample))
                plt.close(fig)

            # learning step
            if epoch < 1 and i_sample <= finetune_treshod:
                lr = init_learning_rate
                nbr_c = init_neighbor_coef
            else:
                lr = finetune_learning_rate
                nbr_c = finetune_neighbor_coef
            for i in range(3):
                if i == i_bmu:
                    nodes.iloc[i] += lr * (sample[:4] - nodes.iloc[i])
                else:
                    nodes.iloc[i] += lr * nbr_c * (sample[:4] - nodes.iloc[i])

            # plot state after (post) step
            if verbose == 2:
                nodes_scores = pca.transform(nodes)

                fig = plt.figure()
                ax = fig.subplots()
                ax.scatter(scores[:, 0], scores[:, 1], c=label_mapper(dataset.iloc[:, 4]), cmap="Set2")
                plt.title("e{:d} - #iter: {:d} - after step".format(epoch, i_sample))
                for i in range(3):
                    if i == i_bmu:
                        color = 'r'
                    else:
                        color = 'black'
                    ax.scatter(nodes_scores[i, 0], nodes_scores[i, 1], c=color)
                ax.scatter(scores[i_sample, 0], scores[i_sample, 1], c='orange', s=40)
                plt.savefig("ex3_outputs/e{:d}_{:d}_post".format(epoch, i_sample))
                plt.close(fig)

    # assign labels to nodes
    node_labels_candidates = [[], [], []]
    for i_sample, sample in dataset.iterrows():
        i_bmu = find_bmu(sample, nodes)
        node_labels_candidates[i_bmu].append(sample[4])
    node_labels = [max(set(node_labels_candidates[0]), key=node_labels_candidates[0].count),
                   max(set(node_labels_candidates[1]), key=node_labels_candidates[1].count),
                   max(set(node_labels_candidates[2]), key=node_labels_candidates[2].count)]
    nodes[4] = pd.Series(node_labels)
    return nodes


def som_predict(dataset, nodes):
    predicted_labels = []
    for i_sample, sample in dataset.iterrows():
        i_bmu = find_bmu(sample, nodes)
        predicted_labels.append(nodes.iloc[i_bmu, 4])
    return predicted_labels


def label_mapper(labels):
    result = []
    for x in labels:
        if x == "Iris-setosa":
            result.append(0)
        elif x == "Iris-virginica":
            result.append(1)
        elif x == "Iris-versicolor":
            result.append(2)
        else:
            result.append(-1)
    return result


def confusion_matrix(y_true, y_pred):
    conf_matrix = np.zeros((3, 3), dtype=int)
    for true, predicted in zip(y_true, y_pred):
        conf_matrix[predicted][true] += 1
    return conf_matrix


def sensitivity(conf_mat):
    result = []
    for i in range(3):
        tp = conf_mat[i][i]
        fn = sum([conf_mat[x][i] for x in range(3) if x != i])
        result.append(tp / (tp + fn))
    return result


def specificity(conf_mat):
    result = []
    for i in range(3):
        tn = sum([conf_mat[y][x] for x in range(3) for y in range(3) if x != i and y != i])
        fp = sum([conf_mat[i][x] for x in range(3) if x != i])
        result.append(tn / (tn + fp))
    return result


def accuracy(conf_mat):
    tp = []
    for i in range(3):
        tp.append(conf_mat[i][i])
    return sum(tp) / sum(sum(conf_mat))


if __name__ == '__main__':
    # load data + normalize + shuffle
    data = pd.read_csv("iris.data", header=None)
    data.iloc[:, :4] = normalize(data.iloc[:, :4])

    # divide into train and test dataset
    train = data.iloc[:40]
    test = data.iloc[40:50]
    train = train.append(data.iloc[50:90])
    test = test.append(data.iloc[90:100])
    train = train.append(data.iloc[100:140])
    test = test.append(data.iloc[140:150])

    # shuffle datasets
    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    nodes = som_fit(train, n_epochs=2, verbose=0, finetune_treshod=25,
                    init_learning_rate=.4, finetune_learning_rate=.2,
                    init_neighbor_coef=.3, finetune_neighbor_coef=.03)
    print(nodes)
    predicted_labels = label_mapper(som_predict(test, nodes))
    true_labels = label_mapper(test.iloc[:, 4])

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print(conf_matrix)
    print("sensitivity:", sensitivity(conf_matrix))
    print("specificity:", specificity(conf_matrix))
    print("accuracy:", accuracy(conf_matrix))





