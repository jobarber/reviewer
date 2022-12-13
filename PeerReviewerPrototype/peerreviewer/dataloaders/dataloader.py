import os
from pathlib import Path

from datasets import load_dataset


def get_dataloader(file_format='csv', **split_dirs):
    """
    Gets a custom dataloader from one or more `split_dirs`.

    Args:
        file_format: (str) One of 'csv', 'text', or 'json', depending on
            the format of the files in the directories.
        split_dirs: ({str: dir_}) The name of each split (e.g., 'train'
            or 'test') mapped to the directory containing the files for
            each split. These directory paths should be somewhere
            within the `peerreviewer.data` directory.

    For example:
        >>> dataset = get_dataloader('csv', train='example_train', test='example_test')

    Returns:
        The loaded custom dataset created from these directories.
    """

    # Get the appropriate paths to train and test files
    peer_reviewer_path = Path(__file__).parent.parent
    data_files = dict()
    for split_name, split_path in split_dirs.items():
        base_path = peer_reviewer_path.joinpath(os.path.join('data', split_path))
        train_filepaths = [os.path.join(base_path, fn) for fn in os.listdir(str(base_path.absolute()))]
        data_files[split_name] = train_filepaths

    # Build the dataset
    dataset = load_dataset(file_format, data_files=data_files)
    return dataset


if __name__ == '__main__':
    dataset = get_dataloader(file_format='csv',
                             train='example_train',
                             test='example_test')
    print(dataset)

    print('TRAIN')
    for batch in dataset['train']:
        print(batch)

    print('TEST')
    for batch in dataset['test']:
        print(batch)
