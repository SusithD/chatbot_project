import json

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def preprocess_data(data):
    questions, answers = [], []
    for item in data:
        questions.append(item['question'])
        if item['annotations'][0]['type'] == 'singleAnswer':
            answers.append(item['annotations'][0]['answer'][0])
        else:
            answers.append(item['annotations'][0]['qaPairs'][0]['answer'][0])
    return questions, answers
