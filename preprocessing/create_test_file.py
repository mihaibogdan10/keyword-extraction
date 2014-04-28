

def create_test_file():
    test_writer = csv.writer(open(TEST_FILE, "w"))
    data = []
    for index, row in enumerate(csv.reader(open(TRAIN_FILE))):
        if index % 100000 == 0:
            print index / 100000
        if index < 5.5 * 10 ** 6:
            pass
        else:
            data.append(row)

    test_writer.writerows(data)
