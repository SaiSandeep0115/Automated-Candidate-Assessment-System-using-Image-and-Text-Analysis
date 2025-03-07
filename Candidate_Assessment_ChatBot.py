import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

warnings.filterwarnings("ignore")

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

class InterviewAssessment:
    def __init__(self):
        self.cv_model = load_model('/content/drive/MyDrive/Colab Files/emotion_detection_25.keras')
        self.picture_size = 128

        self.nlp = spacy.load('en_core_web_md')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.lr_model = load('/content/drive/MyDrive/Colab Files/lr_model.joblib')

        self.questions = {
            'simple': [
                {
                    'question': "What is machine learning?",
                    'original_answer': "Machine learning is a subset of artificial intelligence that involves training algorithms to learn patterns from data and make predictions or decisions without being explicitly programmed."
                },
                {
                    'question': "What is a neural network?",
                    'original_answer': "A neural network is a computational model inspired by the human brain, consisting of interconnected layers of nodes (neurons) that process input data to produce output."
                },
                {
                    'question': "What is the difference between classification and regression?",
                    'original_answer': "Classification is used to predict discrete labels or categories, while regression is used to predict continuous numerical values."
                }
            ],
            'hard': [
                {
                    'question': "Can machines achieve true consciousness or self-awareness?",
                    'original_answer': "Machines can simulate aspects of consciousness, but true consciousness or self-awareness remains a philosophical and scientific debate, as it involves subjective experience."
                },
                {
                    'question': "Explain the backpropagation algorithm in detail.",
                    'original_answer': "Backpropagation is a supervised learning algorithm used to train neural networks by calculating the gradient of the loss function with respect to each weight, and adjusting the weights to minimize the loss."
                },
                {
                    'question': "What are the ethical implications of AI in society?",
                    'original_answer': "The ethical implications of AI include concerns about bias, privacy, job displacement, accountability, and the potential for misuse in surveillance or autonomous weapons."
                }
            ]
        }
        self.current_question = None
        self.original_answer = None

    def get_confidence(self, image_path):
        img = image.load_img(image_path, target_size=(self.picture_size, self.picture_size))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.cv_model.predict(img_array)
        return 'Confident' if prediction[0] < 0.5 else 'Unconfident'

    def preprocess(self, text):
        tokens = nltk.word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in self.stop_words]
        return ' '.join(tokens)

    def get_embedding(self, text):
        doc = self.nlp(text)
        return doc.vector

    def calculate_similarity(self, candidate_embedding, original_embedding):
        return cosine_similarity([candidate_embedding], [original_embedding])[0][0]

    def get_performance(self, candidate_answer):
        original_embedding = self.get_embedding(self.preprocess(self.original_answer))
        candidate_embedding = self.get_embedding(self.preprocess(candidate_answer))

        similarity_score = self.calculate_similarity(candidate_embedding, original_embedding)
        return 'Good' if similarity_score >= 0.7 else 'Poor'

    def get_next_question(self, performance):
        if performance == 'Good':
            question_data = np.random.choice(self.questions['hard'])
        else:
            question_data = np.random.choice(self.questions['simple'])

        self.current_question = question_data['question']
        self.original_answer = question_data['original_answer']
        return self.current_question

class Chatbot:
    def __init__(self):
        self.model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def generate_response(self, prompt):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=500,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def chat_loop():
    assessment = InterviewAssessment()
    chatbot = Chatbot()

    image_path = '/content/drive/MyDrive/Colab Files/730.jpg'
    confidence = assessment.get_confidence(image_path)
    print(f"Chatbot: Welcome! Based on your confidence level, you seem {confidence}.")

    performance = 'Poor'
    question = assessment.get_next_question(performance)
    print(f"Chatbot: Let's start with a question: {question}")
    candidate_answer = input("Your answer: ")

    performance = assessment.get_performance(candidate_answer)
    print(f"Chatbot: Performance on this question is: {performance}")

    print("\nChatbot: Let's continue the conversation!")
    while True:
        question = assessment.get_next_question(performance)
        print(f"Chatbot: Here's your next question: {question}")
        candidate_answer = input("Your answer: ")

        performance = assessment.get_performance(candidate_answer)
        print(f"Chatbot: Performance on this question is: {performance}")

        if performance == 'Good':
            print("Chatbot: Great job! You're doing well. Let's try something more challenging.")
        else:
            print("Chatbot: Let's try a simpler question.")

        if input("Chatbot: Type 'exit' to end the conversation or press Enter to continue: ").lower() == 'exit':
            print("Chatbot: Thank you for the conversation!")
            break

chat_loop()