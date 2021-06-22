import json

def get_synsets():
    with open('imagenet_synset_to_human.json', "r") as f:
        raw_labels_dict = json.load(f)
    labels_dict = dict()

    for i in raw_labels_dict:
        labels_dict[raw_labels_dict[i]["id"]] = (
            int(i),
            raw_labels_dict[i]["label"],
        )
    return labels_dict

print(get_synsets())