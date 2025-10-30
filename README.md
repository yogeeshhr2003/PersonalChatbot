# PersonalChatbot

A comprehensive Python-based chatbot application designed to provide intelligent conversational capabilities with natural language processing and interactive features. This project demonstrates the implementation of a personal assistant chatbot with customizable responses and conversation management.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies & Dependencies](#technologies--dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Components & Modules](#components--modules)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Examples](#examples)
- [Contributing](#contributing)

---

## 🎯 Overview

PersonalChatbot is an intelligent chatbot application built using Python that facilitates natural language understanding and generation. The chatbot is designed to handle user queries, maintain conversation context, and provide relevant responses based on predefined patterns and machine learning models. This project serves as an excellent foundation for building advanced conversational AI systems.

### Key Objectives

- Develop an interactive chatbot capable of understanding natural language inputs
- Implement a pattern-matching system for intelligent response generation
- Create a modular architecture for easy extension and customization
- Provide a user-friendly interface for seamless interaction

---

## ✨ Features

### Core Chatbot Features

1. **Natural Language Processing**: Processes and understands user inputs using NLP techniques
2. **Pattern Matching**: Implements intelligent pattern recognition for query classification
3. **Context Awareness**: Maintains conversation history and context for coherent responses
4. **User Interaction**: Supports interactive conversation with real-time responses
5. **Customizable Responses**: Easy-to-modify response patterns and templates
6. **Error Handling**: Graceful handling of invalid inputs and edge cases
7. **Multi-turn Conversations**: Supports continuous dialogue with context retention
8. **Response Variety**: Provides varied responses to similar user inputs

### Advanced Features

- Intent Recognition: Identifies user intent from natural language queries
- Entity Extraction: Extracts important information from user inputs
- Conversation Logging: Logs conversation history for analysis
- Performance Optimization: Efficient processing of large conversation datasets
- Extensible Architecture: Modular design for adding new features

---

## 📁 Project Structure

```
PersonalChatbot/
│
├── main.py                          # Main entry point of the application
│
├── chatbot/                         # Core chatbot package
│   ├── __init__.py                 # Package initialization
│   ├── chatbot.py                  # Main chatbot class
│   ├── nlp_processor.py            # Natural language processing module
│   ├── response_generator.py       # Response generation engine
│   └── intent_classifier.py        # Intent classification module
│
├── utils/                          # Utility modules
│   ├── __init__.py                # Package initialization
│   ├── text_processing.py         # Text preprocessing utilities
│   ├── data_loader.py             # Data loading utilities
│   └── logger.py                  # Logging configuration
│
├── data/                          # Data directory
│   ├── intents.json              # Intent patterns and responses
│   ├── stopwords.txt             # Stopwords for filtering
│   └── training_data.csv         # Training dataset
│
├── models/                        # Pre-trained models
│   ├── intent_classifier_model.pkl    # Trained classifier model
│   └── vectorizer_model.pkl           # TF-IDF vectorizer
│
├── config/                        # Configuration files
│   ├── config.py                 # Application configuration
│   └── settings.py               # Environment settings
│
├── tests/                         # Test suite
│   ├── __init__.py               # Test package initialization
│   ├── test_chatbot.py           # Chatbot unit tests
│   ├── test_nlp.py               # NLP module tests
│   └── test_response_generator.py # Response generation tests
│
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── README.md                     # Project documentation
└── LICENSE                       # MIT License

```

---

## 🛠️ Technologies & Dependencies

### Core Technologies

- **Python 3.8+**: Programming language
- **NLTK**: Natural Language Toolkit for text processing
- **scikit-learn**: Machine learning library for classification
- **NumPy**: Numerical computing library
- **Pandas**: Data manipulation and analysis

### Required Libraries

```
nltk>=3.6.7
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
python-dotenv>=0.19.0
requests>=2.26.0
```

### Optional Libraries

- **Spacy**: Advanced NLP processing
- **TensorFlow**: Deep learning capabilities
- **Flask**: Web interface deployment
- **PyQt5**: GUI development

---

## 📦 Installation

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

```bash
python --version
```

### Step 1: Clone the Repository

```bash
git clone https://github.com/yogeeshhr2003/PersonalChatbot.git
cd PersonalChatbot
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the virtual environment:

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('stopwords')"
```

### Step 5: Configure Environment Variables

Copy the `.env.example` file to `.env` and update with your settings:

```bash
cp .env.example .env
```

Edit `.env` file with your configuration:

```
DEBUG=True
LOG_LEVEL=INFO
MAX_RESPONSE_LENGTH=500
CONVERSATION_HISTORY_SIZE=50
```

---

## 🚀 Usage

### Basic Usage

Run the chatbot from the command line:

```bash
python main.py
```

### Interactive Mode

Once started, the chatbot will display a prompt where you can type your messages:

```
PersonalChatbot: Hello! I'm your personal assistant. How can I help you today?
User: What is the weather today?
PersonalChatbot: I can help you with weather information. Please specify your location.
User: Exit
PersonalChatbot: Thank you for chatting with me. Goodbye!
```

### Programmatic Usage

```python
from chatbot.chatbot import PersonalChatbot

# Initialize the chatbot
bot = PersonalChatbot()

# Start conversation
response = bot.process_user_input("Hello")
print(response)

# Continue conversation
response = bot.process_user_input("What's your name?")
print(response)
```

### Web Interface (Optional)

To deploy the chatbot with a web interface:

```bash
python -m flask run --config config/config.py
```

Access the web interface at `http://localhost:5000`

---

## ⚙️ Configuration

### Main Configuration File (`config/config.py`)

```python
class Config:
    DEBUG = True
    LOG_LEVEL = 'INFO'
    MAX_RESPONSE_LENGTH = 500
    CONVERSATION_HISTORY_SIZE = 50
    MODEL_PATH = 'models/intent_classifier_model.pkl'
    INTENTS_PATH = 'data/intents.json'
```

### Intent Configuration (`data/intents.json`)

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey", "Good morning"],
      "responses": ["Hello! How can I assist you?", "Hi there! What can I do for you?"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "Goodbye", "See you later", "Farewell"],
      "responses": ["Goodbye! Have a great day!", "See you soon!"]
    }
  ]
}
```

### Environment Variables (`.env`)

```
DEBUG=True
LOG_LEVEL=INFO
MAX_RESPONSE_LENGTH=500
CONVERSATION_HISTORY_SIZE=50
API_KEY=your_api_key_here
ENABLE_LOGGING=True
LOG_FILE_PATH=logs/chatbot.log
```

---

## 🔧 Components & Modules

### 1. **Chatbot Core Module** (`chatbot/chatbot.py`)

The main chatbot class that orchestrates all operations.

**Key Methods:**
- `__init__()`: Initialize the chatbot with loaded models
- `process_user_input(user_input)`: Process and respond to user input
- `load_intents()`: Load intent patterns from JSON
- `generate_response(classified_intent)`: Generate appropriate response
- `update_conversation_history(user_input, bot_response)`: Maintain conversation log

### 2. **NLP Processor Module** (`chatbot/nlp_processor.py`)

Handles all natural language processing tasks.

**Key Functions:**
- `tokenize(text)`: Tokenize input text into words
- `remove_stopwords(tokens)`: Filter out common stopwords
- `lemmatize(tokens)`: Convert words to their base form
- `extract_entities(text)`: Identify important entities
- `calculate_similarity(text1, text2)`: Measure text similarity

### 3. **Intent Classifier Module** (`chatbot/intent_classifier.py`)

Classifies user input into predefined intent categories.

**Key Methods:**
- `train()`: Train the classifier on intent data
- `predict(user_input)`: Classify user input intent
- `get_confidence_score(prediction)`: Return confidence level
- `update_model()`: Update model with new training data

### 4. **Response Generator Module** (`chatbot/response_generator.py`)

Generates contextually appropriate responses.

**Key Methods:**
- `generate_response(intent, context)`: Generate response for given intent
- `apply_template(template, variables)`: Apply variables to response template
- `select_random_response(responses)`: Select random response from pool
- `format_response(raw_response)`: Format response for display

### 5. **Text Processing Utilities** (`utils/text_processing.py`)

Utility functions for text manipulation.

**Key Functions:**
- `preprocess_text(text)`: Clean and normalize text
- `expand_contractions(text)`: Expand contractions (e.g., "don't" → "do not")
- `remove_special_characters(text)`: Clean special characters
- `convert_to_lowercase(text)`: Standardize text case

### 6. **Data Loader** (`utils/data_loader.py`)

Handles loading and processing data files.

**Key Functions:**
- `load_intents_from_json(file_path)`: Load intents from JSON file
- `load_training_data(file_path)`: Load training dataset
- `validate_data_format(data)`: Validate data structure

### 7. **Logger Utility** (`utils/logger.py`)

Centralized logging configuration.

**Key Functions:**
- `get_logger(name)`: Get logger instance
- `setup_logging(log_file)`: Configure logging
- `log_conversation(user_input, bot_response)`: Log conversation

---

## 🏗️ Architecture

### System Architecture Diagram

```
User Input
    ↓
Text Preprocessing (Tokenization, Lemmatization)
    ↓
Intent Classification (ML Model)
    ↓
Context Analysis & Entity Extraction
    ↓
Response Generation Engine
    ↓
Response Formatting & Output
    ↓
User Output
```

### Data Flow

1. **Input Layer**: User provides text input
2. **Processing Layer**: NLP preprocessing and normalization
3. **Classification Layer**: Intent classification using ML
4. **Context Layer**: Conversation context retrieval
5. **Generation Layer**: Response generation based on intent
6. **Output Layer**: Formatted response delivery to user

### Module Interactions

- **main.py** → Initializes and manages chatbot instance
- **chatbot.py** → Orchestrates all core operations
- **nlp_processor.py** → Processes raw user input
- **intent_classifier.py** → Classifies processed input
- **response_generator.py** → Creates appropriate response
- **utils/** → Provides supporting functions

---

## 💡 How It Works

### Step-by-Step Conversation Flow

1. **User Input Reception**
   - User enters text message
   - Input is captured and stored

2. **Text Preprocessing**
   - Tokenization: Split text into words
   - Lowercasing: Standardize case
   - Stopword Removal: Filter common words
   - Lemmatization: Convert to base form

3. **Intent Classification**
   - Pre-processed text is vectorized
   - Machine learning model predicts intent
   - Confidence score is calculated

4. **Response Generation**
   - Intent is identified from classifier
   - Matching response templates are retrieved
   - Context information is incorporated
   - Response is formatted

5. **Output Delivery**
   - Response is displayed to user
   - Conversation is logged
   - History is updated

### Machine Learning Model

- **Algorithm**: Naive Bayes / SVM Classification
- **Vectorization**: TF-IDF
- **Training**: Supervised learning on intent patterns
- **Accuracy**: ~85-95% depending on training data quality

---

## 📚 Examples

### Example 1: Basic Greeting

```python
from chatbot.chatbot import PersonalChatbot

bot = PersonalChatbot()

# Example conversation
print("Bot:", bot.process_user_input("Hello"))
# Output: Bot: Hello! How can I assist you?

print("Bot:", bot.process_user_input("What's your name?"))
# Output: Bot: I'm your Personal Assistant Chatbot. How can I help?
```

### Example 2: Information Query

```python
print("Bot:", bot.process_user_input("Tell me about Python"))
# Output: Bot: Python is a high-level programming language known for its simplicity...

print("Bot:", bot.process_user_input("Can you help with coding?"))
# Output: Bot: Yes, I can assist you with coding questions and programming help.
```

### Example 3: Conversation with Context

```python
response1 = bot.process_user_input("I want to learn programming")
print("Bot:", response1)

response2 = bot.process_user_input("What should I start with?")
print("Bot:", response2)  # Context-aware response about programming

response3 = bot.process_user_input("Thank you!")
print("Bot:", response3)
```

### Example 4: Handling Edge Cases

```python
# Unknown input
print("Bot:", bot.process_user_input("asdfghjkl"))
# Output: Bot: I didn't quite understand that. Could you rephrase?

# Empty input
print("Bot:", bot.process_user_input(""))
# Output: Bot: Please provide input to continue.

# Exit conversation
print("Bot:", bot.process_user_input("Exit"))
# Output: Bot: Goodbye! Thank you for chatting with me.
```

---

## 🧪 Testing

### Running Unit Tests

```bash
python -m pytest tests/
```

### Running Specific Test Suite

```bash
python -m pytest tests/test_chatbot.py -v
python -m pytest tests/test_nlp.py -v
```

### Test Coverage

```bash
python -m pytest --cov=chatbot tests/
```

### Example Test Case

```python
# tests/test_chatbot.py
import unittest
from chatbot.chatbot import PersonalChatbot

class TestPersonalChatbot(unittest.TestCase):
    
    def setUp(self):
        self.bot = PersonalChatbot()
    
    def test_greeting_response(self):
        response = self.bot.process_user_input("Hello")
        self.assertIn("Hello", response)
    
    def test_context_retention(self):
        self.bot.process_user_input("My name is John")
        response = self.bot.process_user_input("What's my name?")
        self.assertIn("John", response)

if __name__ == '__main__':
    unittest.main()
```

---

## 🔐 Security Considerations

- **Input Sanitization**: All user inputs are sanitized to prevent injection attacks
- **Data Privacy**: Conversation logs can be encrypted and securely stored
- **Error Messages**: Generic error messages to avoid information disclosure
- **Rate Limiting**: Optional rate limiting to prevent abuse
- **API Keys**: Sensitive credentials stored in environment variables

---

## 📈 Performance Optimization

- **Model Caching**: Pre-loaded models for faster inference
- **Vectorization**: Efficient text vectorization using TF-IDF
- **Lazy Loading**: Components loaded on-demand
- **Response Pooling**: Pre-computed responses for common queries
- **Conversation History Limit**: Maximum history size to manage memory

---

## 🐛 Troubleshooting

### Common Issues

**Issue: NLTK data not found**
```bash
python -c "import nltk; nltk.download('all')"
```

**Issue: Module import errors**
```bash
pip install --upgrade -r requirements.txt
```

**Issue: Empty responses from chatbot**
- Check if `data/intents.json` is properly configured
- Verify all required models are in `models/` directory

**Issue: Low accuracy in intent classification**
- Increase training data size
- Add more diverse intent patterns
- Retrain the model with updated data

---

## 📝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit changes: `git commit -m 'Add YourFeature'`
4. Push to branch: `git push origin feature/YourFeature`
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Include type hints where applicable

---

## 👥 Author

**yogeeshhr2003**

- GitHub: [yogeeshhr2003](https://github.com/yogeeshhr2003)

---

## 📞 Support

For issues, questions, or suggestions:

- Open an issue on GitHub: [Issues](https://github.com/yogeeshhr2003/PersonalChatbot/issues)
- Check existing documentation and examples
- Review the troubleshooting section above

---

## 🔗 Related Resources

- [NLTK Documentation](https://www.nltk.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Python Natural Language Processing](https://www.coursera.org/learn/python-nlp)
- [Chatbot Development Guide](https://en.wikipedia.org/wiki/Chatbot)

---

## 📊 Project Statistics

- **Language**: Python 100%
- **License**: MIT
- **Status**: Active Development
- **Last Updated**: October 2025

---

## 🎓 Learning Outcomes

By exploring this project, you will learn:

- Natural Language Processing fundamentals
- Machine Learning classification techniques
- Python package development
- Software architecture design patterns
- Testing and debugging practices
- Git version control workflows

---

## 🚀 Future Enhancements

- [ ] Integrate with external APIs (Weather, News, etc.)
- [ ] Add voice input/output capabilities
- [ ] Implement deep learning models (LSTM, Transformer)
- [ ] Create web-based dashboard
- [ ] Add multi-language support
- [ ] Implement sentiment analysis
- [ ] Create mobile application
- [ ] Add machine learning model improvements

---

## ⭐ Acknowledgments

- Thanks to the open-source community
- NLTK and scikit-learn libraries
- Contributors and testers
- Python community for excellent tools and resources

---

**Last Updated**: October 30, 2025

For the latest updates, visit the [GitHub Repository](https://github.com/yogeeshhr2003/PersonalChatbot)
