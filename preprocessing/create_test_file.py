import csv
from utils.module_utils import TRAIN_FILE, TEST_FILE, QUESTIONS_NO, print_overwrite

def create_test_file():
    #use the last 10% of training data to construct a test file
    test_writer = csv.writer(open(TEST_FILE, "w"))
    data = []
    for index, row in enumerate(csv.reader(open(TRAIN_FILE))):
        if index % 10000 == 0:
            print_overwrite("Traversing train file: " + str(index * 100 / QUESTIONS_NO) + " %")
        if index < QUESTIONS_NO * 0.9:
            pass
        else:
            data.append(row)

    test_writer.writerows(data)

if __name__ == "__main__":
    create_test_file()
