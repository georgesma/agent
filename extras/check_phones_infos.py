from lib.dataset_wrapper import Dataset

DATASET_NAME = "pb2007"

dataset = Dataset(DATASET_NAME)

dataset_phones = []
for item_lab in dataset.lab.values():
    for label in item_lab:
        if label["name"] not in dataset_phones:
            dataset_phones.append(label["name"])

print(dataset_phones)

vowels, consonants = dataset.phones_infos["vowels"], dataset.phones_infos["consonants"]
for vowel in vowels:
    assert vowel in dataset_phones
    dataset_phones.remove(vowel)

for consonant in consonants:
    assert consonant in dataset_phones
    dataset_phones.remove(consonant)

print(dataset_phones)
