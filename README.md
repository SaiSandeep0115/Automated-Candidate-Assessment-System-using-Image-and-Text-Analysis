# Automated Candidate Assessment System using Image and Text analysis

## Languages:
- **Python**

## Important Libraries:
- **TensorFlow/Keras**
- **NumPy**
- **Pandas**
- **NLTK**
- **Scikit-learn**
- **Transformers (Hugging Face)**
- **Torch**
- **spaCy**

## Technologies:
- **Computer Vision**
- **Natural Language Processing (NLP)**
- **Large Language Models (LLMs)**
- **Cosine Similarity for Text Comparison**
- **Dynamic Questioning**
- **Performance Evaluation**
- **Emotion Detection**
- **Sentiment Analysis**

## Abstract
This project focuses on building an **Automated Candidate Assessment System** that evaluates candidates during an interview using a combination of **Computer Vision (CV)**, **Natural Language Processing (NLP)**, and **Large Language Models (LLMs)**. The system assesses the candidate's **confidence level** through facial expression analysis, evaluates their **answer performance** by comparing responses to predefined answers, and dynamically adjusts the interview questions based on their performance. The goal is to provide a comprehensive and adaptive chatbot to identify the candidate's strengths and prepares the next question appropriately.

## Dataset
The system uses the following data:
1. **Interview Candidate**: For confidence detection using a custom DL architecture for emotion detection.
2. **Subjective Answers**: Candidate answers and predefined original answers for performance evaluation.
3. **Prompt Question Bank**: A set of simple and hard questions for dynamic questioning.

## Methodology
1. **Computer Vision (CV) Component**:
   - Uses a custom DL architecture to classify the candidate's confidence level as **Confident** or **Unconfident** based on facial expressions.

2. **Natural Language Processing (NLP) Component**:
   - Preprocesses candidate answers using **tokenization**, **lemmatization**, and **stopword removal**.
   - Computes **cosine similarity** between the candidate's answer and the original answer to evaluate performance as **Good** or **Poor**.
   - Additional use of **random forest model** to identify data-driven decision boundary for classifying answers instead of relying completely on arbitrary similarity thresholds.

3. **Large Language Model (LLM) Chatbot**:
   - Uses a pre-trained LLM (**DeepSeek-R1-Distill-Qwen-1.5B**) to conduct the interview.
   - Dynamically adjusts the difficulty of questions based on the candidate's confidence and performance.
   - Provides feedback and complements the candidate when they perform well.

4. **Integration**:
   - Combines CV, NLP, and LLM components into a unified framework.
   - Continuously evaluates the candidate's confidence and performance to adapt the interview flow.

## Models
1. **Computer Vision Model**:
   - Pre-trained custom deep neural network architecture for confidence classification.

2. **NLP and ML Models**:
   - **spaCy** for text embedding.
   - **Cosine Similarity** for answer performance evaluation.
   - **Random Forest model** for classifing decision boundary

3. **LLM Chatbot**:
   - **DeepSeek-R1-Distill-Qwen-1.5B** for generating questions and responses.

## Results and Conclusion
The **Interview Candidate Assessment Framework** successfully integrates **Computer Vision**, **NLP**, and **LLM** technologies to create an adaptive and interactive interview experience. Key outcomes include:
- Classification of candidate confidence using facial expressions with an accuracy of 75%.
- Effective evaluation of answer performance using cosine similarity along with random forest with 99.95% accuracy.
- Dynamic adjustment of question difficulty based on candidate performance.
- Seamless interaction with the LLM chatbot for a natural interview flow.

This framework is to assess candidates more effectively and provide actionable insights for improvement. Future enhancements could include:
- More natural way of conversation with the chatbot
- Expanding the question bank for more diverse topics.
- Incorporating real-time feedback during the interview.

## Disclaimer
This project is designed for demonstration purposes and requires significant computational resources to further research and build a better-performing LLM. Additionally, some components of the project are still incomplete and require modifications, which I am actively working on. The accuracy of confidence detection and performance evaluation depends on the quality of input data and model assumptions. Further testing and validation are recommended before deploying the system in real-world scenarios.
