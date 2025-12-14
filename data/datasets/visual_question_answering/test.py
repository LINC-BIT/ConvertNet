from utils.common.data_record import read_json, write_json
from data.datasets.visual_question_answering.glossary import normalize_word
from collections import defaultdict, Counter
from tqdm import tqdm

ann_data = read_json('/data/zql/datasets/vqav2/Annotations/v2_mscoco_train2014_annotations.json')
question_data = read_json('/data/zql/datasets/vqav2/Questions/v2_OpenEnded_mscoco_train2014_questions.json')



question_to_id = {}

for q in tqdm(question_data['questions']):
    question_to_id[q['question_id']] = q['question']


classes_set = []
for ann in ann_data['annotations']:
    classes_set += [normalize_word(ann['multiple_choice_answer'])]
counter = {k: v for k, v in Counter(classes_set).items() if v >= 9}
ans2label = {k: i for i, k in enumerate(counter.keys())}
label2ans = list(counter.keys())


# print(list(ans2label.keys()))
# exit()

available_classes = list(ans2label.values())
classes_split_1 = available_classes[0: 100]
classes_split_2 = available_classes[100: ]

print(classes_split_1)

dataset_info_1 = [] # (image_file_path, question, labels, scores)
dataset_info_2 = [] # (image_file_path, question, labels, scores)

def get_score(occurences):
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0


ii = 0

pbar = tqdm(ann_data['annotations'])
for q in pbar:
    answers = q["answers"]
    answer_count = {}
    for answer in answers:
        answer_ = answer["answer"]
        answer_count[answer_] = answer_count.get(answer_, 0) + 1

    labels = []
    scores = []
    for answer in answer_count:
        if answer not in ans2label:
            continue
        labels.append(ans2label[answer])
        score = get_score(answer_count[answer])
        scores.append(score)
        
    if len(labels) == 0:
        continue

    # annotations[q["image_id"]][q["question_id"]].append(
    #     {"labels": labels, "scores": scores,}
    # )
    
    # full_label = [0] * len(ans2label)
    # for label_idx, score in zip(labels, scores):
    #     full_label[label_idx] = score
    
    if all([label in classes_split_1 for label in labels]):
        dataset_info_1 += [(q["image_id"], question_to_id[q["question_id"]], labels, scores)]
    elif all([label in classes_split_2 for label in labels]):
        dataset_info_2 += [(q["image_id"], question_to_id[q["question_id"]], [ii - 100 for ii in labels], scores)]
    else:
        # print('ignore')
        pass
    
    # dataset_info += [(q["image_id"], question_to_id[q["question_id"]], labels, scores)]
    
    pbar.set_description(f'# samples: {len(dataset_info_1)}, {len(dataset_info_2)}')
    # print(dataset_info[-1])
    # break
    
    # if ii < 10:
    #     print(dataset_info[-1])
    # ii += 1
    
write_json('/data/zql/datasets/vqav2/label1.json', dataset_info_1)
write_json('/data/zql/datasets/vqav2/label2.json', dataset_info_2)
