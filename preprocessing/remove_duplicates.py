import csv
import hashlib

from utils.module_utils import TRAIN_FILE

def md5(text):
   return int(hashlib.md5(text).hexdigest(), 16) 

def remove_duplicates():
    documents = {}
    exact_duplicates = 0
    title_desc = 0
    duplicates = set()

    for index, line in enumerate(csv.reader(open("Train.csv"))):
        if index % 1000000 == 0:
            print index, "documents processed"

        title = line[1]
        description = line[2]
        keywords = line[3]

        if title not in documents:
            documents[title] = (md5(description), keywords)
        else:
            if md5(description) == documents[title][0] and keywords == documents[title][1]:
                exact_duplicates = exact_duplicates + 1
                duplicates.add(index)
            elif md5(description) == documents[title][0]: 
                title_desc = title_desc + 1
                documents[title] = (documents[title][0], ' '.join(list(set(keywords.split(' ') + \
                                                                documents[title][1].split(' ')))))
                duplicates.add(index)

    writer = csv.writer(open(TRAIN_FILE, "w"), delimiter=',', quoting=csv.QUOTE_ALL)
    for index, line in enumerate(csv.reader(open("Train.csv"))):
        if index not in duplicates:
            title = line[1]
            line[3] = documents[title][1]
            writer.writerow(line)

    print exact_duplicates, " exact duplicates removed"
    print title_desc, "title & description duplicates had their keywords merged"

if __name__ == "__main__":
    remove_duplicates()
