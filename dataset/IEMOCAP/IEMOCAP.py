import torch.utils.data as data


def Partition(ith_fold, DataMap):
    train_map, test_map = {}, {}
    train_data_list, train_length_list, train_labels_emotion_list = [], [], []
    test_map = DataMap['Session{}'.format(ith_fold)]  # data for test
    for i in range(1, 6):
        sess = 'Session{}'.format(i)
        if i != ith_fold:
            train_data_list.extend(DataMap[sess]['data'])
            train_length_list.extend(DataMap[sess]['length'])
            train_labels_emotion_list.extend(DataMap[sess]['labels_emotion'])
    train_map['data'] = train_data_list
    train_map['labels_emotion'] = train_labels_emotion_list
    train_map['length'] = train_length_list
    train_dataset = IEMOCAP(train_map)
    test_dataset = IEMOCAP(test_map)
    return train_dataset, test_dataset


class IEMOCAP(data.Dataset):
    """Speech dataset."""

    def __init__(self, data):
        self.labels_emotion = data['labels_emotion']
        self.data = data['data']
        self.length = data['length']

    def __len__(self):
        return len(self.labels_emotion)

    def __getitem__(self, idx):
        data = self.data[idx]
        length = self.length[idx]
        emo_id = self.labels_emotion[idx]
        return data, length, emo_id