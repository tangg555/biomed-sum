RAKE
====

A Python implementation of the Rapid Automatic Keyword Extraction (RAKE) algorithm as described in: Rose, S., Engel, D., Cramer, N., & Cowley, W. (2010). Automatic Keyword Extraction from Individual Documents. 

This is forked from [aneesha/RAKE](https://github.com/aneesha/RAKE) to support more features :

1. support python 3.0

2. support command line call

3. rearrange the codes to increase the readability

Usage
======

You can easily run it in cmd.

```
python rake.py [-h] [-o OUTPUTFILEPATH] [-m MINPHRASECHAR] [-a MAXPHRASELENGTH] inputFilePath stopwordsFilePath
```

positional arguments:

  inputFilePath         The file path of input document(s). One line represents a document.
  
  stopwordsFilePath     The file path of stopwords, each line represents a word.

optional arguments:

  -h, --help            show this help message and exit
  
  -o OUTPUTFILEPATH, --outputFilePath OUTPUTFILEPATH The file path of output. (default output.txt in current dir).
  
  -m MINPHRASECHAR, --minPhraseChar MINPHRASECHAR The minimum number of characters of a phrase.(default 1)
  
  -a MAXPHRASELENGTH, --maxPhraseLength MAXPHRASELENGTH The maximum length of a phrase.(default 3)


Example
========

```
python rake.py example/input.txt example/stopwords.txt -o example/output.txt
```

License
=======

MIT License