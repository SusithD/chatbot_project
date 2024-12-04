import json
from datasets import Dataset

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def prepare_data(data):
    questions = []
    answers = []
    start_positions = []
    end_positions = []

    for item in data:
        if 'annotations' in item:
            for annotation in item['annotations']:
                print(annotation)  # Add this line to inspect the structure
                annotation_type = annotation.get('type')

                if annotation_type == 'singleAnswer':
                    answer = annotation.get('answer', [])
                    if answer:
                        questions.append(item['question'])
                        answers.append(answer[0])
                        start_positions.append(0)  # Dummy start position
                        end_positions.append(len(answer[0]) - 1)  # Dummy end position
                    else:
                        print(f"Skipping singleAnswer: No answer for question: {item['question']}")
                
                elif annotation_type == 'multipleQAs':
                    for qa in annotation.get('qaPairs', []):
                        qa_question = qa.get('question')
                        qa_answer = qa.get('answer', [])
                        if qa_answer:
                            questions.append(qa_question)
                            answers.append(qa_answer[0])
                            start_positions.append(0)  # Dummy start position
                            end_positions.append(len(qa_answer[0]) - 1)  # Dummy end position
                        else:
                            print(f"Skipping multipleQA: No answer for question: {qa_question}")
    print(f"Prepared {len(questions)} question-answer pairs.")
    return Dataset.from_dict({
        'question': questions,
        'answer': answers,
        'start_positions': start_positions,
        'end_positions': end_positions
    })



if __name__ == "__main__":
    # Load and preprocess the training data
    train_data = load_data("data/train.json")
    print(f"Loaded {len(train_data)} examples from train.json")
    prepared_train_data = prepare_data(train_data)
    prepared_train_data.save_to_disk("data/prepared_train")
    print(f"Saved prepared training data with {len(prepared_train_data)} examples")

    # Load and preprocess the dev data
    dev_data = load_data("data/dev.json")
    print(f"Loaded {len(dev_data)} examples from dev.json")
    prepared_dev_data = prepare_data(dev_data)
    prepared_dev_data.save_to_disk("data/prepared_dev")
    print(f"Saved prepared dev data with {len(prepared_dev_data)} examples")
