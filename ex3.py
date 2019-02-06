import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
                ax.scatter(scores[:, 0], scores[:, 1], c=color_mapper(dataset.iloc[:, 4]), cmap="Set2")
                plt.title("#iter: {:d} - before step".format(i_sample))
                for i in range(3):
                    if i == i_bmu:
                        color = 'r'
                    else:
                        color = 'black'
                    ax.scatter(nodes_scores[i, 0], nodes_scores[i, 1], c=color)
                ax.scatter(scores[i_sample, 0], scores[i_sample, 1], c='orange', s=40)
                plt.savefig("ex3_outputs/{:d}_before".format(i_sample))
                plt.close(fig)

            # learning step
            if i_sample <= finetune_treshod:
                lr = init_learning_rate
                nbr_c = init_neighbor_coef
            else:
                lr = finetune_learning_rate
                nbr_c = finetune_neighbor_coef
            for i in range(3):
                if i == i_bmu:
                    nodes[i] += lr * (sample[:4] - nodes[i])
                else:
                    nodes[i] += lr * nbr_c * (sample[:4] - nodes[i])

            # plot state after (post) step
            if verbose == 2:
                nodes_scores = pca.transform(nodes)

                fig = plt.figure()
                ax = fig.subplots()
                ax.scatter(scores[:, 0], scores[:, 1], c=color_mapper(dataset.iloc[:, 4]), cmap="Set2")
                plt.title("#iter: {:d} - after step".format(i_sample))
                for i in range(3):
                    if i == i_bmu:
                        color = 'r'
                    else:
                        color = 'black'
                    ax.scatter(nodes_scores[i, 0], nodes_scores[i, 1], c=color)
                ax.scatter(scores[i_sample, 0], scores[i_sample, 1], c='orange', s=40)
                plt.savefig("ex3_outputs/{:d}_post".format(i_sample))
                plt.close(fig)

    nodes_scores = pca.transform(nodes)

    fig = plt.figure()
    ax = fig.subplots()
    ax.scatter(scores[:, 0], scores[:, 1], c=color_mapper(dataset.iloc[:, 4]), cmap="Set2")
    ax.scatter(nodes_scores[:, 0], nodes_scores[:, 1], c='black')
    plt.show()
    plt.close(fig)

    # assign labels to nodes
    node_labels_candidates = [[], [], []]
    i = 0
    while i < dataset.shape[0] and (len(node_labels_candidates[0]) < 10 or len(node_labels_candidates[1]) < 10 or len(node_labels_candidates[2]) < 10):
        sample = dataset.iloc[i]
        i_bmu = find_bmu(sample, nodes)
        if len(node_labels_candidates[i_bmu]) < 10:
            node_labels_candidates[i_bmu].append(sample[4])
        i += 1
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


def color_mapper(labels):
    result = []
    for x in labels:
        if x == "Iris-setosa":
            result.append(0)
        elif x == "Iris-virginica":
            result.append(1)
        else:
            result.append(2)
    return result


if __name__ == '__main__':
    # load data
    data = pd.read_csv("iris.data", header=None)

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

    nodes = som_fit(train, verbose=0, finetune_treshod=25,
                    init_learning_rate=.4, finetune_learning_rate=.1,
                    init_neighbor_coef=.3, finetune_neighbor_coef=.01)
    print(nodes)
    predicted_labels = som_predict(test, nodes)
    true_labels = test.iloc[:, 4]

    pca = PCA(2)
    test_scores = pca.fit_transform(test.iloc[:, :4])

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(test_scores[:, 0], test_scores[:, 1], c=color_mapper(predicted_labels), cmap="Set2")
    ax2 = fig.add_subplot(212)
    ax2.scatter(test_scores[:, 0], test_scores[:, 1], c=color_mapper(true_labels), cmap="Set2")
    plt.show()


