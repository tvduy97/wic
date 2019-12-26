import numpy as np
from nltk import word_tokenize
from allennlp.commands.elmo import ElmoEmbedder


def process_data(data_path):
    """Process data: transform text data to vector by ElmoEmbedder.

    Args:
        data_path: data file path.

    Returns:
        wic1, wic2: List of processed data for word in context_1 and word in context_2.
    """
    elmo = ElmoEmbedder()

    positions, context_1, context_2 = np.loadtxt(
        data_path, dtype=str, delimiter='\t', usecols=(2, 3, 4), unpack=True, encoding='utf-8')
    wic1 = []
    wic2 = []
    pic1 = []
    pic2 = []
    for position in positions:
        pos1, pos2 = np.fromstring(position, dtype='int32', sep='-')
        pic1.append(pos1)
        pic2.append(pos2)
    for i in range(len(context_1)):
        context = word_tokenize(context_1[i])
        context = elmo.embed_sentence(context)
        wic1.append(context[2][pic1[i]])
    for i in range(len(context_2)):
        context = word_tokenize(context_2[i])
        context = elmo.embed_sentence(context)
        wic2.append(context[2][pic2[i]])

    return wic1, wic2


def process_label(labels_file_path):
    """Process labels: transform labels from T and F to 0 and 1.

        Args:
            labels_file_path: labels file path.

        Returns:
            labels: List of labels processed to 0 and 1.
        """
    y = np.loadtxt(labels_file_path, dtype=str, unpack=True, encoding='utf-8')
    labels = []
    for label in y:
        if label == 'T':
            labels.append(np.uint8(1))
        else:
            labels.append(np.uint8(0))

    return labels


# Process train data
train_data_path = 'data/train/train.data.txt'
train_labels_path = 'data/train/train.gold.txt'
train_wic1, train_wic2 = process_data(train_data_path)
train_labels = process_label(train_labels_path)

# Process dev data
dev_data_path = 'data/dev/dev.data.txt'
dev_labels_path = 'data/dev/dev.gold.txt'
dev_wic1, dev_wic2 = process_data(dev_data_path)
dev_labels = process_label(dev_labels_path)

# Process test data
test_data_path = 'data/test/test.data.txt'
test_wic1, test_wic2 = process_data(test_data_path)

# Save processed data to file
np.save('processed_data/train_wic1.npy', train_wic1)
np.save('processed_data/train_wic2.npy', train_wic2)
np.save('processed_data/train_labels.npy', train_labels)
np.save('processed_data/dev_wic1.npy', dev_wic1)
np.save('processed_data/dev_wic2.npy', dev_wic2)
np.save('processed_data/dev_labels.npy', dev_labels)
np.save('processed_data/test_wic1.npy', test_wic1)
np.save('processed_data/test_wic2.npy', test_wic2)
