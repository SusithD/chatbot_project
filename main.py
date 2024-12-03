from app.utils import load_data, preprocess_data

train_data = load_data('data/train.json')
questions, answers = preprocess_data(train_data)

print(f"Sample Question: {questions[0]}")
print(f"Sample Answer: {answers[0]}")
