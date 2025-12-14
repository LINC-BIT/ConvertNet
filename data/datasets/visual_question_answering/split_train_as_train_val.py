from utils.common.data_record import read_json, write_json
import random
import copy
from utils.dl.common.env import set_random_seed

set_random_seed(1)


question_data = read_json("/data/zql/datasets/vqav2/Questions/v2_OpenEnded_mscoco_train2014_questions.json")

train_val_split_ratio = 0.8

num_train = int(len(question_data["questions"]) * train_val_split_ratio)

train_question_data = copy.deepcopy(question_data)
val_question_data = copy.deepcopy(question_data)

train_question_data["questions"] = train_question_data["questions"][:num_train]
val_question_data["questions"] = val_question_data["questions"][num_train:]

write_json("/data/zql/datasets/vqav2/Questions/v2_OpenEnded_mscoco_train2014_questions_train.json.my_train_split", train_question_data)
write_json("/data/zql/datasets/vqav2/Questions/v2_OpenEnded_mscoco_train2014_questions_train.json.my_val_split", val_question_data)


