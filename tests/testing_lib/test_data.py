import pytest

from inclearn.lib import data


@pytest.mark.parametrize("dataset_name,increment,n_tasks", [
    ("cifar100", 10, 10),
    ("cifar100", 2, 50)
])
def test_incremental_class(dataset_name, increment, n_tasks):
    dataset = data.IncrementalDataset(
        dataset_name,
        increment=increment
    )

    assert dataset.n_tasks == n_tasks

    current_class = 0
    for _ in range(dataset.n_tasks):
        task_info, train_loader, _, test_loader = dataset.new_task()

        min_c, max_c = current_class, current_class + increment

        assert task_info["increment"] == increment
        assert task_info["min_class"] == min_c
        assert task_info["max_class"] == max_c

        for _, targets, _ in train_loader:
            assert all(min_c <= t.item() < max_c for t in targets)
        for _, targets, _ in test_loader:
            assert all(0 <= t.item() < max_c for t in targets)

        current_class += increment
