import collections
import itertools
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm

from .loops import _print_metrics

logger = logging.getLogger(__name__)


def perclass_loop(
    inc_dataset,
    class_ids,
    devices,
    n_epochs,
    optimizer,
    loss_function,
    task,
    n_tasks,
    target_to_word,
    network=None,
    batch_size=128,
    word_embeddings=None,
    scheduler=None,
    preprocessing=None,
    memory_class_ids=None,
    memory=None,
    disable_progressbar=False,
    features_key="raw_features",
    max_per_batch=3000
):
    if len(devices) > 1:
        logger.info("Duplicating model on {} gpus.".format(len(devices)))
        training_network = nn.DataParallel(network, devices)
    else:
        training_network = network

    visual_features, visual_targets = _extract_features(
        class_ids,
        training_network,
        inc_dataset,
        devices[0],
        memory_class_ids=memory_class_ids,
        memory=memory,
        disable_progressbar=disable_progressbar,
        features_key=features_key
    )

    if preprocessing is not None:
        all_features = torch.cat(list(visual_features.values()))
        logger.info(f"Features shape: {str(all_features.shape)}.")
        preprocessing.fit(all_features)
        del all_features
        for k in list(visual_features.keys()):
            visual_features[k] = preprocessing.transform(visual_features[k])

    # Actually train the generator
    if n_epochs > 0:
        logger.info("Training the generator...")
    for epoch in range(n_epochs):
        metrics = collections.defaultdict(float)

        prog_bar = tqdm(
            class_ids,
            ascii=True,
            bar_format="{desc}: {bar} | {percentage:3.0f}%",
            disable=disable_progressbar
        )

        for batch_index, class_id in enumerate(prog_bar, start=1):
            class_loss = 0.

            qt = max(visual_features[class_id].shape[1] // max_per_batch, 1)
            for i in range(qt):
                lo_index = i * max_per_batch
                hi_index = (i + 1) * max_per_batch

                real_features = visual_features[class_id][lo_index:hi_index]

                optimizer.zero_grad()

                if batch_size is None:
                    words = target_to_word([class_id for _ in range(len(real_features))]
                                          ).to(devices[0])
                else:
                    words = target_to_word([class_id for _ in range(batch_size)]).to(devices[0])
                semantic_features = word_embeddings(words)

                loss = loss_function(real_features, semantic_features, class_id, words, metrics)
                loss.backward()
                optimizer.step()

                class_loss += loss.item()

            metrics["loss"] += class_loss / qt
            _print_metrics(metrics, prog_bar, epoch, n_epochs, batch_index, task, n_tasks)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics["gmm_loss"] / batch_index)
            else:
                scheduler.step(epoch)

    for class_id in class_ids:
        visual_targets[class_id] = torch.cat(visual_targets[class_id])
    return torch.cat(list(visual_features.values())), torch.cat(list(visual_targets.values()))


def linear_loop(
    visual_features,
    visual_targets,
    devices,
    n_epochs,
    optimizer,
    loss_function,
    task,
    n_tasks,
    target_to_word,
    word_embeddings=None,
    scheduler=None,
    batch_size=128,
    normalize=False,
    disable_progressbar=False
):
    loader = _get_loader(visual_features, visual_targets, batch_size=batch_size)

    # Actually train the generator
    if n_epochs > 0:
        logger.info("Training the linear transformation...")
    for epoch in range(n_epochs):
        metrics = collections.defaultdict(float)

        prog_bar = tqdm(
            loader,
            ascii=True,
            bar_format="{desc}: {bar} | {percentage:3.0f}%",
            disable=disable_progressbar
        )

        for batch_index, (x, y) in enumerate(prog_bar, start=1):
            optimizer.zero_grad()

            words = target_to_word(y).to(devices[0])
            semantic_features = word_embeddings(words)

            if normalize:
                loss = loss_function(
                    F.normalize(x, dim=1, p=2), F.normalize(semantic_features, dim=1, p=2)
                )
            else:
                loss = loss_function(x, semantic_features)
            loss.backward()
            optimizer.step()

            metrics["linear_loss"] += loss.item()
            _print_metrics(metrics, prog_bar, epoch, n_epochs, batch_index, task, n_tasks)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics["linear_loss"] / batch_index)
            else:
                scheduler.step(epoch)


def adv_autoencoder_loop(
    inc_dataset,
    class_ids,
    devices,
    n_epochs,
    task,
    n_tasks,
    target_to_word,
    network=None,
    batch_size=128,
    autoencoder=None,
    preprocessing=None,
    memory_class_ids=None,
    memory=None,
    lr=0.0002,
    disable_progressbar=False,
    features_key="raw_features"
):
    if len(devices) > 1:
        logger.info("Duplicating model on {} gpus.".format(len(devices)))
        training_network = nn.DataParallel(network, devices)
    else:
        training_network = network

    visual_features, visual_targets = _extract_features(
        class_ids,
        training_network,
        inc_dataset,
        devices[0],
        memory_class_ids=memory_class_ids,
        memory=memory,
        disable_progressbar=disable_progressbar,
        features_key=features_key
    )
    visual_features = torch.cat(list(visual_features.values()))
    for class_id in class_ids:
        visual_targets[class_id] = torch.cat(visual_targets[class_id])
    visual_targets = torch.cat(list(visual_targets.values()))
    loader = _get_loader(visual_features, visual_targets, batch_size=batch_size)

    if preprocessing is not None:
        assert False
        preprocessing.fit_transform(visual_features)

    optimizer_e = torch.optim.Adam(autoencoder.encoder.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(autoencoder.discriminator.parameters(), lr=lr)
    optimizer_g = torch.optim.Adam(autoencoder.decoder.parameters(), lr=lr)

    for epoch in range(n_epochs):
        metrics = collections.defaultdict(float)

        prog_bar = tqdm(
            loader,
            ascii=True,
            bar_format="{desc}: {bar} | {percentage:3.0f}%",
            disable=disable_progressbar
        )

        for batch_index, (x, y) in enumerate(prog_bar, start=1):
            optimizer_e.zero_grad()
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()

            words = target_to_word(y).to(devices[0])
            attributes = autoencoder.emb(words)
            noise = autoencoder.get_noise(len(x))

            # Training encoder-decoder:
            pred_noise = autoencoder.encoder(x)
            reconstruction = autoencoder.decoder(torch.cat((attributes, pred_noise), dim=-1))
            mse_loss = F.mse_loss(reconstruction, x)
            mse_loss.backward()

            optimizer_e.step()
            optimizer_g.step()

            # Training discriminator:
            autoencoder.encoder.eval()
            pred_noise = autoencoder.encoder(x)
            fake_dis = autoencoder.discriminator(pred_noise)
            true_dis = autoencoder.discriminator(noise)

            dis_true_loss = F.binary_cross_entropy_with_logits(
                true_dis,
                torch.ones_like(true_dis).to(devices[0])
            )
            dis_fake_loss = F.binary_cross_entropy_with_logits(
                fake_dis,
                torch.zeros_like(fake_dis).to(devices[0])
            )
            dis_loss = dis_true_loss + dis_fake_loss
            dis_loss.backward()
            optimizer_d.step()

            # Training generator:
            optimizer_g.zero_grad()
            autoencoder.encoder.train()
            pred_noise = autoencoder.encoder(x)
            fake_dis = autoencoder.discriminator(pred_noise)
            gen_loss = F.binary_cross_entropy_with_logits(
                fake_dis,
                torch.ones_like(fake_dis).to(devices[0])
            )
            gen_loss.backward()
            optimizer_g.step()

            metrics["rec"] += mse_loss.item()
            metrics["dis"] += dis_loss.item()
            metrics["gen"] += gen_loss.item()
            _print_metrics(metrics, prog_bar, epoch, n_epochs, batch_index, task, n_tasks)

    return visual_features, visual_targets


def features_to_classifier_loop(
    features,
    targets,
    flags,
    epochs,
    optimizer,
    classifier,
    loss_function,
    scheduler=None,
    disable_progressbar=False
):
    loader = _get_loader(features, targets, flags)

    for epoch in range(epochs):
        metrics = collections.defaultdict(float)

        prog_bar = tqdm(
            loader,
            ascii=True,
            bar_format="{desc}: {bar} | {percentage:3.0f}%",
            disable=disable_progressbar
        )

        for batch_index, (x, y, f) in enumerate(prog_bar, start=1):
            optimizer.zero_grad()
            logits = classifier(x)["logits"]
            loss = loss_function(logits, y, f, metrics)
            loss.backward()
            optimizer.step()

            metrics["loss"] += loss.item()
            _print_metrics(metrics, prog_bar, epoch, epochs, batch_index, 0, 1)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics["loss"] / batch_index)
            else:
                scheduler.step(epoch)


def online_generation(
    features,
    targets,
    epochs,
    optimizer,
    classifier,
    loss_function,
    word_embeddings,
    target_to_word,
    unseen_class_ids=None,
    scheduler=None,
    unseen_amount=100,
    disable_progressbar=False
):

    word_embeddings.eval()
    for epoch in range(epochs):
        metrics = collections.defaultdict(float)

        fake_features, fake_targets = [], []
        for class_id in unseen_class_ids:
            class_ids = [class_id for _ in range(unseen_amount)]
            words = target_to_word(class_ids).to(word_embeddings.device)

            with torch.no_grad():
                fake_features.append(word_embeddings(words))
            fake_targets.append(torch.tensor(class_ids).to(word_embeddings.device))
        fake_features = torch.cat(fake_features)
        fake_targets = torch.cat(fake_targets)

        loader = _get_loader(
            torch.cat((features, fake_features), dim=0),
            torch.cat((targets.to(word_embeddings.device), fake_targets), dim=0)
        )
        prog_bar = tqdm(
            loader,
            ascii=True,
            bar_format="{desc}: {bar} | {percentage:3.0f}%",
            disable=disable_progressbar
        )

        for batch_index, (x, y) in enumerate(prog_bar, start=1):
            optimizer.zero_grad()

            logits = classifier(x)["logits"]
            loss = loss_function(logits, y)
            loss.backward()

            optimizer.step()

            metrics["loss"] += loss.item()
            _print_metrics(metrics, prog_bar, epoch, epochs, batch_index, 0, 1)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics["loss"] / batch_index)
            else:
                scheduler.step(epoch)

    word_embeddings.train()


# ---------
# Utilities
# ---------


def _extract_features(
    class_ids,
    training_network,
    inc_dataset,
    device=None,
    memory_class_ids=None,
    memory=None,
    disable_progressbar=False,
    features_key="raw_features"
):
    """Extract features for every given class, and keep them in GPU memory for
    faster loading.

    :param class_ids: List of classes to extract.
    :param training_network: The network used to extract.
    :param inc_dataset: The incremental dataset needed to fetch data.
    :param device: A potential GPU device.
    :param memory_class_ids: List of old classes that belong to memory.
    :param memory: A tuple of (data_x, data_y), in numpy format.
    :param disable_progressbar: Hide progress bar, useful for gridsearch.

    :return: A tuple of both dict class_id->features and targets.
    """
    if memory_class_ids is None:
        memory_class_ids = []  # Shouldn't set empty list as default value, google it.

    # Extract features
    visual_features = collections.defaultdict(list)
    visual_targets = collections.defaultdict(list)

    logger.info("Computing class features...")
    prog_bar = tqdm(
        class_ids, ascii=True, bar_format="{bar} | {percentage:3.0f}%", disable=disable_progressbar
    )
    for index, class_id in enumerate(prog_bar, start=1):
        if class_id in memory_class_ids:
            # We cannot extract all features, the class is "old", and thus only
            # memory data is available for it.
            class_memory = _select_memory(memory, class_id)
            loader_args = [[]]
            loader_kwargs = {"memory": class_memory, "mode": "test", "data_source": "train"}
        else:
            # New class, thus everything can be used, enjoy!
            loader_args = [[class_id]]
            loader_kwargs = {"mode": "test", "data_source": "train"}

        loader_base = inc_dataset.get_custom_loader(*loader_args, **loader_kwargs)[1]
        loader_kwargs["mode"] = "flip"
        loader_flip = inc_dataset.get_custom_loader(*loader_args, **loader_kwargs)[1]

        for loader in (loader_base, loader_flip):
            for input_dict in loader:
                inputs, targets = input_dict["inputs"], input_dict["targets"]
                with torch.no_grad():
                    outputs = training_network(inputs.to(device))

                visual_features[class_id].append(outputs[features_key])
                visual_targets[class_id].append(targets)

        visual_features[class_id] = torch.cat(visual_features[class_id])

    return visual_features, visual_targets


def _select_memory(memory, class_id):
    mem_x, mem_y = memory
    indexes = np.where(mem_y == class_id)[0]
    return mem_x[indexes], mem_y[indexes]


def _get_loader(features, targets, flags=None, batch_size=128):

    class Dataset(torch.utils.data.Dataset):

        def __init__(self, features, targets, flags=None):
            self.features = features
            self.targets = targets
            self.flags = flags

        def __len__(self):
            return self.features.shape[0]

        def __getitem__(self, index):
            f, t = self.features[index], self.targets[index]
            if self.flags is None:
                return f, t, 1
            return f, t, self.flags[index]

    dataset = Dataset(features, targets, flags)
    return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
