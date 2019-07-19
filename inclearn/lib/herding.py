import numpy as np

from inclearn.lib import utils


def closest_to_mean(features, nb_examplars):
    features = features / (np.linalg.norm(features, axis=0) + 1e-8)
    class_mean = np.mean(features, axis=0)

    return _l2_distance(features, class_mean).argsort()[:nb_examplars]


def icarl_selection(features, nb_examplars):
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0],))

    w_t = mu
    iter_herding, iter_herding_eff = 0, 0

    while not (
        np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])
    ) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    herding_matrix[np.where(herding_matrix == 0)[0]] = 10000

    return herding_matrix.argsort()[:nb_examplars]


def minimize_confusion(inc_dataset, network, memory, class_index, nb_examplars):
    _, new_loader = inc_dataset.get_custom_loader(class_index, mode="test")
    new_features, _ = utils.extract_features(network, new_loader)
    new_mean = np.mean(new_features, axis=0)

    from sklearn.cluster import KMeans

    n_clusters = 4
    model = KMeans(n_clusters=n_clusters)
    model.fit(new_features)

    indexes = []
    for i in range(n_clusters):
        cluster = model.cluster_centers_[i]
        distances = _l2_distance(cluster, new_features)

        indexes.append(distances.argsort()[:nb_examplars // n_clusters])

    return np.concatenate(indexes)


    if memory is None:
        # First task
        #return icarl_selection(new_features, nb_examplars)
        return np.random.permutation(new_features.shape[0])[:nb_examplars]

    distances = _l2_distance(new_mean, new_features)

    data_memory, targets_memory = memory
    for indexes in _split_memory_per_class(targets_memory):
        _, old_loader = inc_dataset.get_custom_loader(
            [],
            memory=(data_memory[indexes], targets_memory[indexes]),
            mode="test"
        )

        old_features, _ = utils.extract_features(network, old_loader)
        old_mean = np.mean(old_features, axis=0)

        # The larger the distance to old mean
        distances -= _l2_distance(old_mean, new_features)

    return distances.argsort()[:int(nb_examplars)]

# ---------
# Utilities
# ---------


def _l2_distance(x, y):
    return np.power(x - y, 2).sum(-1)


def _split_memory_per_class(targets):
    max_class = max(targets)

    for class_index in range(max_class):
        yield np.where(targets == class_index)[0]
