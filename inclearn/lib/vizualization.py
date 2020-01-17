import torch


def grad_cam(spatial_features, selected_logits):
    batch_size = spatial_features.shape[0]
    assert batch_size == len(selected_logits)

    formated_logits = [selected_logits[i] for i in range(batch_size)]

    import pdb
    pdb.set_trace()
    grads = torch.autograd.grad(
        formated_logits, spatial_features, retain_graph=True, create_graph=True
    )

    assert grads.shape == spatial_features.shape

    return grads
