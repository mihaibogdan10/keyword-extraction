from positive_class_classifier import PositiveClassClassifier
from utils.module_utils import TRAIN_FILE, iter_documents, iter_minibatches

if __name__ == "__main__": 
    length = 0

    minibatch_iterator = iter_minibatches(TRAIN_FILE, PositiveClassClassifier.hvectorizer)
    for batch_no, (x_title_train, x_description_train, y_train) in enumerate(minibatch_iterator):
        length += sum(len(y) for y in y_train)

    print length
