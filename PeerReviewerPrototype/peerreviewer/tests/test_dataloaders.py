from peerreviewer.dataloaders.dataloader import get_dataloader


def test_train_csv_dataloader():
    dataset = get_dataloader(file_format='csv',
                             train='example_test',
                             test='example_train')

    for batch in dataset['train']:
        assert {'sentence': 'The movie was fun to watch!', 'label': 1} == batch
        break


def test_test_csv_dataloader():
    dataset = get_dataloader(file_format='csv',
                             train='example_test',
                             test='example_train')

    for batch in dataset['test']:
        assert {'sentence': 'The movie was great!', 'label': 1} == batch
        break
