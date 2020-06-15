import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.nn import functional as F

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


def random(features, nb_examplars):
    return np.random.permutation(len(features))[:nb_examplars]


def kmeans(features, nb_examplars, k=5):
    """Samples examplars for memory according to KMeans.

    :param features: The image features of a single class.
    :param nb_examplars: Number of images to keep.
    :param k: Number of clusters for KMeans algo, defaults to 5
    :return: A numpy array of indexes.
    """
    model = KMeans(n_clusters=k)
    cluster_assignements = model.fit_predict(features)

    nb_per_clusters = nb_examplars // k
    indexes = []
    for c in range(k):
        c_indexes = np.random.choice(np.where(cluster_assignements == c)[0], size=nb_per_clusters)
        indexes.append(c_indexes)

    return np.concatenate(indexes)


def confusion(ypreds, ytrue, nb_examplars, class_id=None, minimize_confusion=True):
    """Samples examplars for memory according to the predictions.

    :param ypreds: All the predictions (shape [b, c]).
    :param ytrue: The true label.
    :param nb_examplars: Number of images to keep.
    :param minimize_confusion: Samples easiest examples or hardest.
    """
    indexes = np.where(ytrue == class_id)[0]
    ypreds, ytrue = ypreds[indexes], ytrue[indexes]

    ranks = ypreds.argsort(axis=1)[:, ::-1][np.arange(len(ypreds)), ytrue]

    indexes = ranks.argsort()
    if minimize_confusion:
        return indexes[:nb_examplars]
    return indexes[-nb_examplars:]


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
            [], memory=(data_memory[indexes], targets_memory[indexes]), mode="test"
        )

        old_features, _ = utils.extract_features(network, old_loader)
        old_mean = np.mean(old_features, axis=0)

        # The larger the distance to old mean
        distances -= _l2_distance(old_mean, new_features)

    return distances.argsort()[:int(nb_examplars)]


def var_ratio(memory_per_class, network, loader, select="max", type=None):
    var_ratios = []
    for input_dict in loader:
        inputs = input_dict["inputs"].to(network.device)
        with torch.no_grad():
            outputs = network(inputs)
        var_ratios.append(outputs["var_ratio"])
    var_ratios = np.concatenate(var_ratios)

    indexes = var_ratios.argsort()
    if select == "max":
        return indexes[-memory_per_class:]
    elif select == "min":
        return indexes[:memory_per_class]
    raise ValueError("Only possible value for <select> are [max, min], not {}.".format(select))


def mcbn(memory_per_class, network, loader, select="max", nb_samples=100, type=None):
    if not hasattr(network.convnet, "sampling_mode"):
        raise ValueError("Network must be MCBN-compatible.")
    network.convnet.sampling_mode()

    all_probs = []
    for input_dict in loader:
        inputs = input_dict["inputs"].to(network.device)

        probs = []
        for _ in range(nb_samples):
            with torch.no_grad():
                outputs = network(inputs)
                logits = outputs["logits"]
                probs.append(F.softmax(logits, dim=-1).cpu().numpy())

        probs = np.stack(probs)
        all_probs.append(probs)
    network.convnet.normal_mode()

    all_probs = np.concatenate(all_probs, axis=1)
    var_ratios = _var_ratio(all_probs.transpose(1, 0, 2))

    indexes = var_ratios.argsort()
    assert len(indexes) == all_probs.shape[1]
    if select == "max":
        return indexes[-memory_per_class:]
    elif select == "min":
        return indexes[:memory_per_class]
    raise ValueError("Only possible value for <select> are [max, min], not {}.".format(select))


# ---------
# Utilities
# ---------


def _var_ratio(sampled_probs):
    predicted_class = sampled_probs.max(axis=2)

    hist = np.array(
        [
            np.histogram(predicted_class[i, :], range=(0, 10))[0]
            for i in range(predicted_class.shape[0])
        ]
    )

    return 1. - hist.max(axis=1) / sampled_probs.shape[1]


def _l2_distance(x, y):
    return np.power(x - y, 2).sum(-1)


def _split_memory_per_class(targets):
    max_class = max(targets)

    for class_index in range(max_class):
        yield np.where(targets == class_index)[0]
