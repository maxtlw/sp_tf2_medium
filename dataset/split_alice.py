import pathlib
import re

import nltk

""" 
    Split Alice in Wonderland into sentences, removing additional undesired information. 
    Taken from https://gist.github.com/phillipj/4944029.
"""

ALICE_FILE_PATH = pathlib.Path(__file__).parent.joinpath('alice_in_wonderland.txt')
SPLITTED_ALICE_FILE_PATH = pathlib.Path(__file__).parent.joinpath('splitted_alice_in_wonderland.txt')

with open(ALICE_FILE_PATH, encoding='utf-8') as infile:
    whole_text = infile.read()

# Remove single newlines (they are just due to the source text formatting)
whole_text = re.sub(r'\n([^\n])', r' \1', whole_text)
# Replace multiple leftover newlines with a single one
whole_text = re.sub(r'\n{2,}', r'\n', whole_text)
# Remove multiple spaces
whole_text = re.sub(r' +', r' ', whole_text)
# Remove * characters (probably inserted to divide chapters)
whole_text = re.sub(r'\*( )*', '', whole_text)
# Replace ` (begin of a quote) with ', just for simplicity
whole_text = whole_text.replace('`', "'")

splitted_text = nltk.sent_tokenize(whole_text, language='english')

with open(SPLITTED_ALICE_FILE_PATH, 'w', encoding='utf-8') as outfile:
    # Need some further preprocessing to remove unwanted pieces (ugly, but it works pretty fine!)
    prev_was_chapter = False
    written_lines = 0
    for sentence in splitted_text:
        splitted_sentence = sentence.split('\n')
        for sentence_piece in splitted_sentence:
            clean_sentence_piece = sentence_piece.strip()
            if 'CHAPTER' in clean_sentence_piece:
                prev_was_chapter = True
                continue
            if 'THE END' in clean_sentence_piece or \
                    'Lewis Carroll' in clean_sentence_piece or \
                    'adventures in wonderland' in clean_sentence_piece.lower() or \
                    'FULCRUM EDITION' in clean_sentence_piece:
                continue
            if prev_was_chapter:
                prev_was_chapter = False
                continue
            outfile.write(clean_sentence_piece + '\n')
            written_lines += 1

print(f'{written_lines} lines written into {SPLITTED_ALICE_FILE_PATH}.')



