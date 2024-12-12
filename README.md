# **Python Programming Chatbot**
![Gradient Midnight Presentation - Opening](https://github.com/user-attachments/assets/c1e7a3b0-749b-42c4-98d3-b636d564808e)
![Screenshot 2024-12-12 231226](https://github.com/user-attachments/assets/63af786a-7c15-4487-b4b3-204540908567)
![Screenshot 2024-12-12 231258](https://github.com/user-attachments/assets/a44b2340-cc05-4e15-ac4c-d3e5796a084f)
![Screenshot 2024-12-12 231341](https://github.com/user-attachments/assets/33db55ae-a172-4b48-8d8d-dfa25fbeb55e)

Welcome to the Python Programming Chatbot project repository! This project is a robust and intelligent chatbot designed to assist Python developers and learners with real-time solutions to programming challenges. By leveraging cutting-edge natural language processing models, the chatbot delivers accurate, context-aware, and actionable responses to coding queries.

---

## **Table of Contents**

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Technologies Used](#technologies-used)  
4. [Dataset](#dataset)  
5. [Model Training and Fine-Tuning](#model-training-and-fine-tuning)  
6. [Applications](#applications)  
7. [Project Structure](#project-structure)  
8. [Installation and Usage](#installation-and-usage)  
9. [Future Work](#future-work)  
10. [Contributing](#contributing)  
11. [License](#license)  

---

## **1. Project Overview**

The Python Programming Chatbot is developed to assist users in solving Python-related queries. It is fine-tuned on Salesforce's `codegen-350M-multi` model and uses a custom dataset of real-world Python challenges. This chatbot can be deployed in educational platforms, developer tools, and coding assistants.

---

## **2. Features**

- **Interactive Python Programming Assistance:** Responds to Python programming queries with tailored solutions.
- **Real-World Problem Solving:** Handles real-world coding scenarios, including debugging, optimization, and scripting.
- **Developer-Friendly Interface:** Seamless integration for developers needing real-time coding support.
- **Scalable Backend:** Built using Flask for API development and MongoDB for chat history storage.

---

## **3. Technologies Used**

### **Programming Languages**  
- Python  

### **Libraries and Frameworks**  
- Flask  
- Hugging Face Transformers  
- Pandas  
- NumPy  
- Scikit-learn  

### **Machine Learning Models**  
- Base Model: Salesforce's `codegen-350M-multi`  
- Fine-tuned Model: Optimized for Python coding dialogues  

### **Database**  
- MongoDB (For chat history storage)  

---

## **4. Dataset**

The custom dataset for this project includes Python programming challenges in a question-answer format.  

### **Dataset Structure**  
- **Instruction:** Describes the task or query.  
- **Input:** Provides additional context.  
- **Output:** Contains the expected Python code solution.  

### **Preprocessing Steps**  
1. Combined `Instruction` and `Input` into a single dialogue format:  
   ```  
   User: [Instruction + Input]  
   Chatbot: [Output]  
   ```  
2. Split into 80% training and 20% evaluation subsets.  
3. Converted to Hugging Face Dataset format.  

---

## **5. Model Training and Fine-Tuning**

The chatbot model was fine-tuned using the Hugging Face Trainer API.

### **Training Parameters**  
- **Batch Size:** 4  
- **Learning Rate:** 5e-5  
- **Epochs:** 3  
- **Gradient Accumulation Steps:** 8  

### **Code Example**
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()
```

---

## **6. Applications**

### **Personalized Learning Platforms**  
Enhances e-learning by providing detailed programming solutions and personalized guidance for Python learners.  

### **Developer Support Systems**  
Assists developers with debugging, error resolution, and best practice suggestions in real-time.  

### **Automated Coding Assistants**  
Boosts productivity by offering quick responses to Python coding challenges, saving time on searching and troubleshooting.

---

## **7. Project Structure**

```
├── dataset/
│   ├── train.json
│   ├── eval.json
├── model/
│   ├── fine_tuned_model/
│   ├── tokenizer/
├── api/
│   ├── app.py
│   ├── requirements.txt
├── README.md
```

- **dataset/**: Contains training and evaluation datasets.  
- **model/**: Stores the fine-tuned model and tokenizer.  
- **api/**: Flask-based API files for chatbot interaction.  

---

## **8. Installation and Usage**

### **Prerequisites**  
- Python 3.8 or higher  
- MongoDB  

### **Steps**  
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/python-chatbot.git
   cd python-chatbot
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r api/requirements.txt
   ```

3. **Set Up the Database**  
   - Install and configure MongoDB.  
   - Update the connection string in `app.py`.  

4. **Run the Flask App**  
   ```bash
   python api/app.py
   ```

5. **Interact with the Chatbot**  
   - Use a REST client like Postman to send queries to the chatbot API.  

---

## **9. Future Work**

- Expand dataset to include more programming languages.  
- Implement a web-based front end for easier user interaction.  
- Enhance model capabilities for handling advanced coding tasks.  

---

## **10. Contributing**

We welcome contributions!  
1. Fork the repository.  
2. Create a new branch:  
   ```bash
   git checkout -b feature-name
   ```  
3. Make changes and commit:  
   ```bash
   git commit -m "Add feature-name"
   ```  
4. Push the branch and open a pull request.

---

## **11. License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
