import language_tool_python
import spacy
import nltk
import math
from transformers import pipeline
from nltk.corpus import reuters
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

multiplier_gr = 3
multiplier_ld = 7
multiplier_sc = 10
multiplier_wf = 1

def round_to_nearest_half(number):
    return round(number * 2) / 2

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
    grammar_score = ((num_sentences - num_errors) / num_sentences)
    rounded_gr = round_to_nearest_half(grammar_score * 9)
    print("Grammatical Range: ", rounded_gr,)
    return rounded_gr

def calculate_ld(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        paragraph = file.read()
    tokens = nltk.word_tokenize(paragraph)
    types = set(tokens)
    lexical_diversity = (len(types) / len(tokens))
    rounded_ld = round_to_nearest_half(lexical_diversity * 9)
    print("Lexical Diversity: ", rounded_ld,)
    return rounded_ld

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
    syntactic_complexity = (complex_sentences / len(list(doc.sents)))
    rounded_sc = round_to_nearest_half(syntactic_complexity * 9)
    print("Syntactic Complexity: ", rounded_sc,)
    return rounded_sc

fdist = FreqDist(word.lower() for word in reuters.words())

def calculate_wf(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        paragraph = file.read()
    tokens = word_tokenize(paragraph)
    score = 0
    for token in tokens:
        # We add 1 to the frequency to avoid division by zero for words not in the corpus
        # We take the square root of the frequency instead of the inverse
        score += 1 / (math.sqrt(fdist[token.lower()]) + 1)
    # Normalize the score to a range of 0 to 9
    word_frequency = (score / len(tokens))
    rounded_wf = round_to_nearest_half(word_frequency * 9)
    print("Word Frequency: ", rounded_wf,)
    return rounded_wf
    
def calculate_weighted_average(score_gr, score_ld, score_wf, score_sc):
    # Calculate the weighted scores
    weighted_score_gr = score_gr * multiplier_gr
    weighted_score_ld = score_ld * multiplier_ld
    weighted_score_wf = score_wf * multiplier_wf
    weighted_score_sc = score_sc * multiplier_sc

    # Calculate the final score
    final_score = weighted_score_gr + weighted_score_ld + weighted_score_wf + weighted_score_sc
    sum_multiplier = multiplier_gr + multiplier_ld + multiplier_wf + multiplier_sc
    average_score = round_to_nearest_half(final_score / sum_multiplier)

    print("Average Weighted Score: ", average_score)
    return average_score

calculate_ta("entry.txt")
rounded_gr = calculate_gr("entry.txt")
rounded_ld = calculate_ld("entry.txt")
rounded_sc = calculate_sc("entry.txt")
rounded_wf = calculate_wf("entry.txt")
calculate_weighted_average(rounded_gr, rounded_ld, rounded_wf, rounded_sc)
