import random
import os
os.environ['NLTK_DATA'] = os.getcwd() + '/nltk_data'
from textblob import TextBlob
from textblob import Word
from config import *

GREETINGS = ("hi", "hello", "how are you", "what's up", "sup")

def is_greeting(sentence):
    for word in sentence.words:
        if(word.lower() in GREETINGS):
            return True
    return False

def about_user(word):
    return word.lower() in FIRST_PERSON_PRONOUNS

def about_bot(word):
    return word.lower() in SECOND_PERSON_PRONOUNS

def is_curse_word(word):
    return word.lower() in FILTER_WORDS

def about_other_person(word):
    return word.lower() in THIRD_PERSON_PRONOUNS

def sort_word(word, pos, nouns, verbs, adjectives, others):
    if(pos.startswith('NN')):
        nouns.append(word)
    elif(pos.startswith('VB')):
        lemma = Word(word).lemmatize('v')
        if(lemma==u'be'):
            verbs.append("are")
        else:
            verbs.append(word)
    elif(pos.startswith('JJ')):
        adjectives.append(word)
    else:
        others.append(word)

def construct_response(pronouns, nouns, verbs, adjectives, other):
    total_words = len(pronouns) + len(nouns) + len(verbs) + len(adjectives) + len(other)
    response_words = []
    while(len(response_words) != total_words):
        if(len(pronouns) != 0):
            response_words.append(pronouns.pop(0))
        if(len(verbs) != 0):
            response_words.append(verbs.pop(0))
        if(len(other) != 0):
            response_words.append(other.pop(0))
        if(len(adjectives) != 0):
            response_words.append(adjectives.pop(0))
        if(len(nouns) != 0):
            response_words.append(nouns.pop(0))

    return " ".join(response_words)

def parse_sentence(sentence):
    nouns = []
    verbs = []
    adjectives = []
    pronouns = []
    parsed = TextBlob(sentence)

    if(is_greeting(parsed)):
        return random.choice(GREETINGS)

    others = []

    for word, pos in parsed.pos_tags:
        if(about_user(word)):
            pronouns.append(random.choice(SECOND_PERSON_PRONOUNS))
        elif(about_bot(word)):
            pronouns.append(random.choice(FIRST_PERSON_PRONOUNS))
        elif(about_other_person(word)):
            pronouns.append(random.choice(THIRD_PERSON_PRONOUNS))
        elif(not is_curse_word(word)):
            sort_word(word, pos, nouns, verbs, adjectives, others)
    return construct_response(pronouns, nouns, verbs, adjectives, others)
    

def main():
    print "Bot started"
    while(True):
        sentence = raw_input()
        print parse_sentence(sentence)

main()


