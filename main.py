import language_tool_python
import spacy
import nltk
from transformers import pipeline
from nltk.corpus import reuters
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

def calculate_ta(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        paragraph = file.read()
        
    # Count words
    words = paragraph.split()
    num_words = len(words)
    
    # Identify the topic of the paragraph
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    topics = ["politics", "sports", "technology", "entertainment", "business", "environment", "food"]  # Define your own topics
    result = classifier(paragraph, topics)
    topic = result["labels"][0]

    # Print the word count and topic in the terminal
    print("Word Count: ", num_words)
    print("Topic: ", topic)

def calculate_gr(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        paragraph = file.read()
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(paragraph)
    sentences = nltk.sent_tokenize(paragraph)
    num_sentences = len(sentences)
    num_errors = len(matches)
    grammar_score = ((num_sentences - num_errors) / num_sentences) * 9

    # Print the grammar score in the terminal
    print("Grammatical Range: ", grammar_score,)

def calculate_ld(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        paragraph = file.read()
    tokens = nltk.word_tokenize(paragraph)
    types = set(tokens)
    lexical_diversity = (len(types) / len(tokens)) * 9

    # Print the lexical diversity score in the terminal
    print("Lexical Diversity: ", lexical_diversity)

def calculate_sc(file_path):
    # Load the SpaCy model
    nlp = spacy.load('en_core_web_sm')
    
    # Read the paragraph from the file
    with open(file_path, 'r') as file:
        paragraph = file.read()
    
    # Parse the paragraph with SpaCy
    doc = nlp(paragraph)
    
    # Initialize the count of sentences that contain at least one of the specified characters
    complex_sentences = 0
    
    # Iterate over each sentence
    for sentence in doc.sents:
        # Check if the sentence contains at least one of the specified characters
        if any(token.text in [',', ';', ':', '-'] for token in sentence):
            complex_sentences += 1
    
    # Calculate the syntactic complexity
    syntactic_complexity = (complex_sentences / len(list(doc.sents))) * 9
    
    print("Syntactic Complexity: ", syntactic_complexity)

fdist = FreqDist(word.lower() for word in reuters.words())

def calculate_wf(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        paragraph = file.read()
    tokens = word_tokenize(paragraph)
    score = 0
    for token in tokens:
        # We add 1 to the frequency to avoid division by zero for words not in the corpus
        score += 1 / (fdist[token.lower()] + 1)
    # Normalize the score to a range of 0 to 100
    word_frequency = (score / len(tokens)) * 9
    print("Word Frequency: ", word_frequency)

calculate_ta("entry.txt")
calculate_gr("entry.txt")
calculate_ld("entry.txt")
calculate_sc("entry.txt")
calculate_wf("entry.txt")