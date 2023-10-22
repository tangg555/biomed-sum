import re
import operator
import argparse
import codecs
import os
from pathlib import Path

def isNumber(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False

class RakeDoc:
    def __init__(self, inputFilePath, stopwordsFilePath, outputFilePath, minPhraseChar, maxPhraseLength):
        self.outputFilePath = outputFilePath
        self.minPhraseChar = minPhraseChar
        self.maxPhraseLength = maxPhraseLength
        # read documents
        self.docs = []
        for document in codecs.open(inputFilePath, 'r', 'utf-8'):
            self.docs.append(document)
        # read stopwords
        stopwords = []
        for word in codecs.open(stopwordsFilePath, 'r', 'utf-8'):
            stopwords.append(word.strip())
        stopwordsRegex = []
        for word in stopwords:
            regex = r'\b' + word + r'(?![\w-])'
            stopwordsRegex.append(regex)
        self.stopwordsPattern = re.compile('|'.join(stopwordsRegex), re.IGNORECASE)

    def separateWords(self, text):
        splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
        words = []
        for word in splitter.split(text):
            word = word.strip().lower()
            # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
            if len(word) > 0 and word != '' and not isNumber(word):
                words.append(word)
        return words
    
    
    def calculatePhraseScore(self, phrases):
        # calculate wordFrequency and wordDegree
        wordFrequency = {}
        wordDegree = {}
        for phrase in phrases:
            wordList = self.separateWords(phrase)
            wordListLength = len(wordList)
            wordListDegree = wordListLength - 1
            for word in wordList:
                wordFrequency.setdefault(word, 0)
                wordFrequency[word] += 1
                wordDegree.setdefault(word, 0)
                wordDegree[word] += wordListDegree
        for item in wordFrequency:
            wordDegree[item] = wordDegree[item] + wordFrequency[item]
    
        # calculate wordScore = wordDegree(w)/wordFrequency(w)
        wordScore = {}
        for item in wordFrequency:
            wordScore.setdefault(item, 0)
            wordScore[item] = wordDegree[item] * 1.0 / wordFrequency[item]

        # calculate phraseScore
        phraseScore = {}
        for phrase in phrases:
            phraseScore.setdefault(phrase, 0)
            wordList = self.separateWords(phrase)
            candidateScore = 0
            for word in wordList:
                candidateScore += wordScore[word]
            phraseScore[phrase] = candidateScore
        return phraseScore
    
        
    def execute(self):
        file = codecs.open(self.outputFilePath,'w','utf-8')
        for document in self.docs:
            # split a document into sentences
            sentenceDelimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
            sentences = sentenceDelimiters.split(document)
            # generate all valid phrases
            phrases = []
            for s in sentences:
                tmp = re.sub(self.stopwordsPattern, '|', s.strip())
                phrasesOfSentence = tmp.split("|")
                for phrase in phrasesOfSentence:
                    phrase = phrase.strip().lower()
                    if phrase != "" and len(phrase) >= self.minPhraseChar and len(phrase.split()) <= self.maxPhraseLength:
                        phrases.append(phrase)
    
            # calculate phrase score
            phraseScore = self.calculatePhraseScore(phrases)
            keywords = sorted(phraseScore.items(), key = operator.itemgetter(1), reverse=True)
            file.write(str(keywords[0:int(len(keywords)/3)]) + "\n")
        file.close()

class Rake(object):
    def __init__(self, stopwords_filepath, min_phrase_char_len: int=1, max_phrase_len: int=3):
        self.min_phrase_char_len = min_phrase_char_len
        self.max_phrase_len = max_phrase_len
        # read stopwords
        stopwords_ = []
        if not Path(stopwords_filepath).exists():
            from nltk.corpus import stopwords
            stopwords_ = stopwords.words('english')
        else:
            for word in codecs.open(stopwords_filepath, 'r', 'utf-8'):
                stopwords_.append(word.strip())
        stopwords_regex = []
        for word in stopwords_:
            regex = r'\b' + word + r'(?![\w-])'
            stopwords_regex.append(regex)
        self.stopwords_pattern = re.compile('|'.join(stopwords_regex), re.IGNORECASE)

    def separate_words(self, text):
        splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
        words = []
        for word in splitter.split(text):
            word = word.strip().lower()
            # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
            if len(word) > 0 and word != '' and not isNumber(word):
                words.append(word)
        return words

    def get_phrase_score(self, phrases):
        # calculate wordFrequency and wordDegree
        wordFrequency = {}
        wordDegree = {}
        for phrase in phrases:
            wordList = self.separate_words(phrase)
            wordListLength = len(wordList)
            wordListDegree = wordListLength - 1
            for word in wordList:
                wordFrequency.setdefault(word, 0)
                wordFrequency[word] += 1
                wordDegree.setdefault(word, 0)
                wordDegree[word] += wordListDegree
        for item in wordFrequency:
            wordDegree[item] = wordDegree[item] + wordFrequency[item]

        # calculate wordScore = wordDegree(w)/wordFrequency(w)
        wordScore = {}
        for item in wordFrequency:
            wordScore.setdefault(item, 0)
            wordScore[item] = wordDegree[item] * 1.0 / wordFrequency[item]

        # calculate phraseScore
        phraseScore = {}
        for phrase in phrases:
            phraseScore.setdefault(phrase, 0)
            wordList = self.separate_words(phrase)
            candidateScore = 0
            for word in wordList:
                candidateScore += wordScore[word]
            phraseScore[phrase] = candidateScore
        return phraseScore

    def run(self, input_doc: str):
        # split a document into sentences
        sentence_delimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
        sentences = sentence_delimiters.split(input_doc)
        # generate all valid phrases
        phrases = []
        for s in sentences:
            tmp = re.sub(self.stopwords_pattern, '|', s.strip())
            phrases_of_sentence = tmp.split("|")
            for phrase in phrases_of_sentence:
                phrase = phrase.strip().lower()
                if phrase != "" and len(phrase) >= self.min_phrase_char_len and len(
                        phrase.split()) <= self.max_phrase_len:
                    phrases.append(phrase)
        # calculate phrase score
        phraseScore = self.get_phrase_score(phrases)
        keyword_scores = sorted(phraseScore.items(), key=operator.itemgetter(1), reverse=True)
        keywords = [phrase[0] for phrase in keyword_scores]
        return keywords

if __name__ == '__main__':
    def readParamsFromCmd():
        parser = argparse.ArgumentParser(description = "This is a python implementation of rake(rapid automatic keyword extraction).")
        parser.add_argument('inputFilePath', help = 'The file path of input document(s). One line represents a document.')
        parser.add_argument('stopwordsFilePath', help = 'The file path of stopwords, each line represents a word.')
        parser.add_argument('-o', '--outputFilePath', help = 'The file path of output. (default output.txt in current dir).', default = 'output.txt')
        parser.add_argument('-m', '--minPhraseChar', type = int, help = 'The minimum number of characters of a phrase.(default 1)', default = 1)
        parser.add_argument('-a', '--maxPhraseLength', type = int, help = 'The maximum length of a phrase.(default 3)', default = 3)
        return parser.parse_args()

    params = readParamsFromCmd().__dict__

    rake = RakeDoc(params['inputFilePath'], params['stopwordsFilePath'], params['outputFilePath'], params['minPhraseChar'], params['maxPhraseLength'])
    rake.execute()