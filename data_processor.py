import help_functions

import pandas as pd
import re

import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex
from spacy.tokenizer import Tokenizer

def setup_spacy_parser():
    SPACY_PARSER = spacy.load('en', disable=['spacy_parser', 'ner'])

    #prefix_re = re.compile(r'''^[]''')
    #suffix_re = re.compile(r'''[]$''')
    #infix_re = re.compile(r'''''')

    # modify tokenizer infix patterns
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
    )

    infix_re = compile_infix_regex(infixes)

    SPACY_PARSER.tokenizer.infix_finditer = infix_re.finditer
    SPACY_PARSER.tokenizer.add_special_case("``", [{"ORTH": "``"}])
    SPACY_PARSER.tokenizer.add_special_case("´´", [{"ORTH": "´´"}])
    #SPACY_PARSER.tokenizer.prefix_search = prefix_re.search
    #SPACY_PARSER.tokenizer.suffix_search = suffix_re.search
    return SPACY_PARSER

def fix_period_spaces_and_word_pos(text, word_index):
    splitted_text = text.split()
    new_text = []
    new_word_index = word_index
    for index, word in enumerate(splitted_text):
        re_match = re.search(r'[\w.]+\.(?!\w)', word)
        if re_match:
            new_text.append(word[:-1])
            new_text.append('.')
            if index < word_index:
                new_word_index += 1
        else:
            new_text.append(word)
    return (' ').join(new_text), new_word_index

def fix_quotations_and_word_pos(text, word_index):
    splitted_text = text.split()
    new_text = []
    new_word_index = word_index
    for index, word in enumerate(splitted_text):
        re_match = re.search(r'[`.´.]+(?=\w)', word)
        if re_match:
            new_text.append(re_match.group(0))
            new_text.append(word[re_match.span()[1]:])
            if index < word_index:
                new_word_index += 1
        else:
            new_text.append(word)
    return (' ').join(new_text), new_word_index

def get_lemmatized_text(spacy_parser, text):
    doc = spacy_parser(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text

class DataProcessor:
    """Handles processing of data."""

    def __init__(self, text_data, lemma_data, word_pos_data, sense_key_data, lemmatized_text_data=None, sense_encoded_text_data=None, sensed_lemma_data=None):
        self.data = pd.DataFrame({'text': text_data, 'lemma': lemma_data, 'word_pos': word_pos_data, 'sense_key': sense_key_data})
        self.parser = setup_spacy_parser()

        if lemmatized_text_data:
            self.data["lemmatized_text"] = lemmatized_text_data
        if sense_encoded_text_data:
            self.data["sense_encoded_text"] = sense_encoded_text_data
        if sensed_lemma_data:
            self.data["sensed_lemma"] = sensed_lemma_data

    def get_data(self):
        return self.data

    def fix_text_and_word_index_in_data(self, fixer_function):
        new_text = []
        new_word_index = []
        for _, row in self.data.iterrows():
            new_text_elem, new_word_index_elem = fixer_function(row.text, row.word_pos)
            new_text.append(new_text_elem)
            new_word_index.append(new_word_index_elem)

        self.data.text = new_text
        self.data.word_pos = new_word_index 


    def fix_period_spaces_and_word_index_in_data(self):
        self.fix_text_and_word_index_in_data(fix_period_spaces_and_word_pos)

    def fix_quotations_and_word_index_in_data(self):
        self.fix_text_and_word_index_in_data(fix_quotations_and_word_pos)

    def lemmatize_text_in_data(self):
        self.data['lemmatized_text'] = self.data.text.map(lambda x: get_lemmatized_text(self.parser, x))

    def sense_encode_text_in_data(self, text_col):
        sense_dict = help_functions.build_sense_dict(self.data.lemma.to_list(), self.data.sense_key.to_list())
        def get_sensed_lemma(row):
            return str(row.lemma[:-2]+'_'+str(sense_dict[row.lemma][row.sense_key]))
        def get_new_text(row):
            words = row[text_col].split(' ')
            new_text = words[:row.word_pos]+[row.sensed_lemma]+words[row.word_pos+1:]
            return " ".join(new_text)

        self.data['sensed_lemma'] = self.data.apply(get_sensed_lemma, axis=1)
        self.data['sense_encoded_text'] = self.data.apply(get_new_text, axis=1)
        