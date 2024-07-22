"""
Copyright (C) 2024  Gigas64

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You will find a copy of this licenseIn the root directory of the project
or you can visit <https://www.gnu.org/licenses/> For further information.
"""

import inspect
from io import StringIO
import re
import pyarrow as pa
import pyarrow.compute as pc
import spacy
from markdown import Markdown
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from spacy.language import Language
from torch import cuda, backends
from nn_rag.components.commons import Commons
from nn_rag.intent.abstract_knowledge_intent import AbstractKnowledgeIntentModel
from tqdm.auto import tqdm


class KnowledgeIntent(AbstractKnowledgeIntentModel):
    """This class represents RAG intent actions whereby data preparation can be done
    """

    EOL_CHARACTERS = [
        '\n\n',  # Double newline, sometimes used for paragraph breaks
        '\n',  # Unix/Linux/MacOS newline
        '\r\n',  # Windows newline
        '\r',  # Old MacOS newline
    ]

    def filter_on_condition(self, canonical: pa.Table, header: str, condition: list, mask_null: bool=None,
                            include_score: bool=None, disable_progress_bar: bool=None, save_intent: bool=None, 
                            intent_order: int=None, intent_level: [int, str]=None, replace_intent: bool=None, 
                            remove_duplicates: bool=None) -> pa.Table:
        """ Taking a canonical with a text column and removes the rows based on a condition.

        The condition is a list of triple tuples in the form: [(comparison, operation, logic)] where comparison
        is the item or column to compare, the operation is what to do when comparing and the logic if you are
        chaining tuples as in the logic to join to the next boolean flags to the current. An example might be:

                [(comparison, operation, logic)]
                [(1, 'greater', 'or'), (-1, 'less', None)]
                [(pa.array(['INACTIVE', 'PENDING']), 'is_in', None)]

        The operator and logic are taken from pyarrow compute and are:

                operator => match_substring, match_substring_regex, equal, greater, less, greater_equal, less_equal, not_equal, is_in, is_null
                logic => and, or, xor, and_not

        :param canonical: a pa.Table as the reference table
        :param header: the header for the target values to change
        :param condition: a list of tuple or tuples in the form [(comparison, operation, logic)]
        :param mask_null: (optional) if nulls in the other they require a value representation.
        :param include_score: (optional) if the score should be calculated. This helps with speed. Default is True
        :param disable_progress_bar: (optional) turn the progress bar off and on. Default is False
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the column name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: an equal length list of correlated values
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        canonical = self._get_canonical(canonical)
        header = self._extract_value(header)
        h_col = canonical.column(header).combine_chunks()
        mask = self._extract_mask(h_col, condition=condition, mask_null=mask_null)
        canonical = canonical.filter(mask)
        # Convert the text and rebuild
        text = canonical['text'].to_pylist()
        result = self._build_statistics(text, include_score=include_score, disable_progress_bar=disable_progress_bar)
        return pa.Table.from_pylist(result)

    def filter_on_mask(self, canonical: pa.Table, indices: list=None, pattern: str=None, include_score: bool=None, 
                       disable_progress_bar: bool=None, save_intent: bool=None, intent_level: [int, str]=None, 
                       intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ Taking a canonical with a text column and removes based on either a regex
        pattern or list of index.

        'indices' takes a list of index to be removed, or/and tuples of start add stop
        range of index numbers. For example [1, 3, (5, 8)] would remove the index
        [1, 3, 5, 6, 7].

        'pattern' takes a regex str to find within the text from which that row is
        removed. For example '^Do Not Use Without Permission' would remove rows
        where the text starts with that string.

        :param canonical: a pa.Table as the reference table
        :param indices: (optional) a list of numbers and/or tuples for rows to be dropped
        :param pattern: (optional) a regex expression pattern to remove an element
        :param include_score: (optional) if the score should be calculated. This helps with speed. Default is True
        :param disable_progress_bar: (optional) turn the progress bar off and on. Default is False
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the intent name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        canonical = self._get_canonical(canonical)
        indices = Commons.list_formatter(indices)
        # by pattern
        if pattern:
            # Get the specified column
            column = canonical['text']
            # Compute the regex match for each element in the column
            matches = pc.match_substring_regex(column, pattern)
            # Find the indices where the pattern matches
            indices = [i for i, match in enumerate(matches) if match.as_py()]
        # by indices
        if indices:
            index_list = []
            # expand the indices
            for item in indices:
                if isinstance(item, tuple) and len(item) == 2:
                    start, end = item
                    index_list.extend(range(start, end))
                else:
                    index_list.append(item)
            indices = sorted(list(set(index_list)), reverse=True)
            # Convert the text and rebuild
            text = canonical['text'].to_pylist()
            # Create a new dictionary excluding the specified indices
            text = [value for i, value in enumerate(text) if i not in indices]
            result = self._build_statistics(text, include_score=include_score,
                                            disable_progress_bar=disable_progress_bar)
            return pa.Table.from_pylist(result)
        return canonical

    def replace_on_pattern(self, canonical: pa.Table, pattern: str=None, replacement: str=None,
                           max_replacements: int=None, include_score: bool=None, disable_progress_bar: bool=None,
                           save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                           replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ given as pattern or list of patterns, uses regular expression to find and replace in the text.
        If max_replacements is given and not equal to -1, it limits the maximum amount replacements per input,
        counted from the left. Null values emit null.

        If no pattern is given, the default behavior is to tidy the text by removing end of line characters
        and removing superfluous spaces.

        :param canonical: a pa.Table as the reference table
        :param pattern: (optional) RE2 Regular Expression pattern.
        :param replacement: (optional) replace the pattern with.
        :param max_replacements: (optional) The maximum number of strings to replace in each input value.
        :param include_score: (optional) if the score should be calculated. This helps with speed. Default is True
        :param disable_progress_bar: (optional) turn the progress bar off and on. Default is False
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the intent name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # code block
        canonical = self._get_canonical(canonical)
        text = canonical.column('text')
        if isinstance(pattern, str):
            # make the list an 'or' pattern
            replacement = replacement if isinstance(replacement, str) else ' '
            new_text = pc.replace_substring_regex(text, pattern, replacement, max_replacements=max_replacements)
        else:
            # default behaviour
            eol_pattern = '|'.join(map(re.escape, self.EOL_CHARACTERS))
            eol_text = pc.replace_substring_regex(text, eol_pattern, ' ')
            new_text = pc.replace_substring_regex(eol_text, '\s{2,}', ' ')
        result = self._build_statistics(new_text.to_pylist(), include_score=include_score, disable_progress_bar=disable_progress_bar)
        return pa.Table.from_pylist(result)

    def filter_on_join(self, canonical: pa.Table, indices: list=None, chunk_size: int=None, include_score: bool=None, 
                       disable_progress_bar: bool=None, save_intent: bool=None, intent_level: [int, str]=None,
                       intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ Takes a list of indices and joins those indices with the next row as a sum of the two. This allows
        two rows with high similarity scores to be joined together.

        :param canonical: a pa.Table as the reference table
        :param indices: (optional) a list of index values to be joined to the following row
        :param chunk_size: (optional) tries to optimize the size of the chunks by joining small chunks. Default is 0
        :param include_score: (optional) if the score should be calculated. This helps with speed. Default is True
        :param disable_progress_bar: (optional) turn the progress bar off and on. Default is False
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the intent name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        """
        # intent recipie options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # code block

        def join_strings_at_indices(strings, idx_list):
            if not strings or not idx_list:
                return strings
            idx_list = sorted(idx_list)
            rtn = []
            i = 0
            while i < len(strings):
                if i in idx_list:
                    joined_string = strings[i]
                    while i + 1 < len(strings) and i + 1 in idx_list:
                        joined_string += " " + strings[i + 1]
                        i += 1
                    joined_string += " " + strings[i + 1] if i + 1 < len(strings) else ""
                    rtn.append(joined_string)
                    i += 1
                else:
                    rtn.append(strings[i])
                i += 1
            return rtn

        def join_strings_to_limit(strings, limit):
            rtn = []
            current_string = ""
            for string in strings:
                if len(current_string) + len(string) + (1 if current_string else 0) <= limit:
                    if current_string:
                        current_string += " "
                    current_string += string
                else:
                    rtn.append(current_string)
                    current_string = string
            if current_string:
                rtn.append(current_string)
            return rtn

        canonical = self._get_canonical(canonical)
        indices = Commons.list_formatter(indices)
        chunk_size = chunk_size if isinstance(chunk_size, int) else 0
        text = join_strings_at_indices(canonical['text'].to_pylist(), indices)
        if chunk_size > 0:
            text = join_strings_to_limit(text, chunk_size)
        result = self._build_statistics(text, include_score=include_score, disable_progress_bar=disable_progress_bar)
        return pa.Table.from_pylist(result)

    def text_from_markdown(self, canonical: pa.Table, include_score: bool=None, disable_progress_bar: bool=None,
                           save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                           replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ a Markdown text and converts it to a plain text string

        :param canonical: a pa.Table as the reference table
        :param include_score: (optional) if the score should be calculated. This helps with speed. Default is True
        :param disable_progress_bar: (optional) turn the progress bar off and on. Default is False
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the intent name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        """
        # intent recipie options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # code block

        def unmark_element(element, stream=None):
            if stream is None:
                stream = StringIO()
            if element.text:
                stream.write(element.text)
            for sub in element:
                unmark_element(sub, stream)
            if element.tail:
                stream.write(element.tail)
            return stream.getvalue()

        # patching Markdown
        Markdown.output_formats["plain"] = unmark_element
        _md = Markdown(output_format="plain")
        _md.stripTopLevelTags = False

        canonical = self._get_canonical(canonical)
        text = canonical.column('text').to_pylist()
        eol_pattern = '|'.join(map(re.escape, self.EOL_CHARACTERS))
        plain_text = []
        for item in text:
            plain_text.append(_md.convert(item))
        result = self._build_statistics(plain_text, include_score=include_score, disable_progress_bar=disable_progress_bar)
        return pa.Table.from_pylist(result)

    def text_to_document(self, canonical: pa.Table, sep: str=None, save_intent: bool=None,
                         intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None, 
                         remove_duplicates: bool=None) -> pa.Table:
        """ Takes a table and joins all the row text into a single row.

        :param canonical: a pa.Table as the reference table
        :param sep: (optional) seperator between joining chunks. The default is '\n'
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the intent name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        """
        # intent recipie options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # code block
        canonical = self._get_canonical(canonical)
        sep = sep if isinstance(sep, str) else ' '
        text = canonical.column('text').to_pylist()
        text = [sep.join(text)]
        result = self._build_statistics(text, include_score=False, disable_progress_bar=True)
        return pa.Table.from_pylist(result)

    def text_to_paragraphs(self, canonical: pa.Table, sep: str=None, pattern: str=None, max_char_size: int=None,
                           include_score: bool=None, disable_progress_bar: bool=None, save_intent: bool=None,
                           intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                           remove_duplicates: bool=None) -> pa.Table:
        """ Takes a table with the text column and split it into perceived paragraphs. This method
        is generally used for text discovery and manipulation before chunking.

        :param canonical: a pa.Table as the reference table
        :param sep: (optional) a string to separate the paragraphs. Default is '\n\n'
        :param pattern: (optional) a regex to separate the paragraphs. Must start with a capital or left punctuation
        :param max_char_size: (optional) the maximum number of characters to process at one time
        :param include_score: (optional) if the score should be calculated. This helps with speed. Default is True
        :param disable_progress_bar: (optional) turn the progress bar off and on. Default is False
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the intent name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)

        @Language.component("set_custom_paragraph")
        def set_custom_paragraph(document):
            for token in tqdm(document[:-2], disable=disable_progress_bar, total=len(document)-2, desc='building paragraphs'):
                # Define sentence start if pipe + titlecase token
                if token.text == "|" and (document[token.i + 1].is_title or
                                          document[token.i + 1].is_left_punct):
                    document[token.i + 1].is_sent_start = True
                else:
                    # Explicitly set sentence start to False otherwise, to tell
                    # the parser to leave those tokens alone
                    document[token.i + 1].is_sent_start = False
            return document

        canonical = self._get_canonical(canonical)
        sep = self._extract_value(sep)
        sep = sep if isinstance(sep, str) else '\n\n'
        max_char_size = max_char_size if isinstance(max_char_size, int) else 900_000
        bi_encoder = "multi-qa-mpnet-base-cos-v1"
        # set device
        device = "cuda" if cuda.is_available() else "mps" if hasattr(backends, "mps") and backends.mps.is_available() else "cpu"
        text = canonical.column('text').to_pylist()
        # mark paragraphs with a special separator
        if isinstance(pattern, str):
            prep_text = []
            for chunk in text:
                sub_text = []
                parts = re.split(pattern, chunk)
                matches = re.findall(pattern, chunk)
                elements = [parts[0].strip()] if parts[0] else []
                for i in range(len(matches)):
                    elements.append(str(matches[i] + parts[i + 1]).strip())
                sub_text += elements
                sub_text = ' | '.join(sub_text)
                prep_text.append(sub_text)
            text = prep_text
        else:
            text = [chunk.replace(sep, ' | ') for chunk in text]
        # if text is too large spacy will fail
        text = self._limit_elements(text, 999_990)
        # load English parser
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("set_custom_paragraph", before="parser")
        text_parts = []
        for item in text:
            section = []
            doc = nlp(item)
            for section in doc.sents:
                section = str(section.text).replace(' |', ' ').replace('\n', ' ').strip()
                text_parts.append(str(section))
        text = text_parts
        result = self._build_statistics(text, include_score=include_score, disable_progress_bar=disable_progress_bar)
        return pa.Table.from_pylist(result)

    def text_to_sentences(self, canonical: pa.Table, include_score: bool=None, char_limit: int=None,
                          disable_progress_bar: bool=None, save_intent: bool=None,
                          intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                          remove_duplicates: bool=None) -> pa.Table:
        """ Taking a Table with a text column, returning the profile of that text as a list of sentences. This method
        is generally used for text discovery and manipulation before chunking.

        :param canonical: a pa.Table as the reference table
        :param include_score: (optional) if the score should be included. This helps with speed. Default is True
        :param char_limit: (optional) the maximum number of characters to limit a sentence to after which it is chunked
        :param disable_progress_bar: (optional) turn the progress bar off and on. Default is False
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the intent name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        @Language.component("set_custom_sentence")
        def set_custom_sentence(document):
            for token in tqdm(document[:-1], disable=disable_progress_bar, total=len(document)-1, desc='building sentences'):
                if token in ['.','?','!'] and document[token.i + 1].is_title:
                    document[token.i + 1].is_sent_start = True
            return document

        canonical = self._get_canonical(canonical)
        char_limit = char_limit if isinstance(char_limit, int) else 900_000
        bi_encoder = "multi-qa-mpnet-base-cos-v1"
        # set device
        device = "cuda" if cuda.is_available() else "mps" if hasattr(backends, "mps") and backends.mps.is_available() else "cpu"
        text = canonical.column('text').to_pylist()
        # if text is too large spacy will fail
        text = self._limit_elements(text, 999_990)
        # load English parser
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("set_custom_sentence", before="parser")
        text_parts = []
        for item in text:
            section = []
            doc = nlp(item)
            for section in doc.sents:
                text_parts.append(str(section))
        text = text_parts
        if isinstance(char_limit, int):
            text = self._limit_elements(text, char_limit)
        result = self._build_statistics(text, include_score=include_score, disable_progress_bar=disable_progress_bar)
        return pa.Table.from_pylist(result)

    def text_to_chunks(self, canonical: pa.Table, chunk_size: int=None, overlap: int=None, include_score: bool=None,
                       disable_progress_bar: bool=None, save_intent: bool=None, intent_level: [int, str]=None,
                       intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ Taking a profile Table and converts the sentences into chunks ready for embedding. By default,
        the sentences are joined and then chunked according to the chunk_size. However, if the temperature is used
        the sentences are grouped by temperature and then chunked. Be aware you may get small chunks for
        small sentences.

        :param canonical: a pa.Table as the reference table
        :param chunk_size: (optional) The number of characters per chunk. Default is 500
        :param overlap: (optional) the number of chars a chunk should overlap. Note this adds to the size of the chunk
        :param include_score: (optional) if the score should be included. This helps with speed. Default is True
        :param disable_progress_bar: (optional) turn the progress bar off and on. Default is False
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the intent name that groups intent to create a column
        :param intent_order: (optional) the order in which each intent should run.
                    - If None: default's to -1
                    - if -1: added to a level above any current instance of the intent section, level 0 if not found
                    - if int: added to the level specified, overwriting any that already exist

        :param replace_intent: (optional) if the intent method exists at the level, or default level
                    - True - replaces the current intent method with the new
                    - False - leaves it untouched, disregarding the new intent

        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params

        def split_strings(strings, chunk_length, overlap_length):
            rtn_list = []
            for string in tqdm(strings, disable=disable_progress_bar, total=len(strings), desc='building chunks'):
                i = 0
                while i < len(string):
                    start = max(0,i) if i == 0 else  max(0, i - overlap_length)
                    end = i + chunk_length + overlap_length if i == 0 else i + chunk_length
                    rtn_list.append(string[start:end])
                    i += chunk_length
            return rtn_list

        canonical = self._get_canonical(canonical)
        chunk_size = self._extract_value(chunk_size)
        chunk_size = chunk_size if isinstance(chunk_size, int) else 512
        overlap = self._extract_value(overlap)
        overlap = overlap if isinstance(overlap, int) else int(chunk_size / 10)
        text = canonical.column('text').to_pylist()
        text = split_strings(text, chunk_size - overlap, overlap)
        result = self._build_statistics(text, include_score=include_score, disable_progress_bar=disable_progress_bar)
        return pa.Table.from_pylist(result)

    #  ---------
    #   Private
    #  ---------

    @staticmethod
    def _limit_elements(strings: list, limit: int):
        rtn_list = []
        for i, string in enumerate(strings):
            if len(string) > limit:
                start = 0
                while start < len(string):
                    end = min(start + limit, len(string))
                    rtn_list.append(string[start:end])
                    start = end
            else:
                rtn_list.append(string)
        return rtn_list

    @staticmethod
    def _build_statistics(text: list, include_score: bool, disable_progress_bar: bool):
        """ Builds the statistical view of a list of text"""
        include_score = include_score if isinstance(include_score, bool) else False
        disable_progress_bar = disable_progress_bar if isinstance(disable_progress_bar, str) else False
        # if text is too large spacy will fail
        result = []
        for num, section in enumerate(text):

            result.append({
                "index": num,
                "char_count": len(section),
                "word_count": len(section.split(" ")),
                "sentence_count": len(section.split(". ")),
                "token_count": round(len(section) / 4),  # 1 token = ~4 chars, see:
                "score": 0,
                'text': section,
            })
        if include_score:
            device = "cuda" if cuda.is_available() else "mps" if hasattr(backends,"mps") and backends.mps.is_available() else "cpu"
            text_list = [item['text'] for item in result]
            l1 = text_list[:-1]
            l2 = text_list[1:]
            model = CrossEncoder("cross-encoder/stsb-roberta-base", device=device)
            sentence_pairs = [[l1, l2] for l1, l2 in zip(l1, l2)]
            ce_scores = model.predict(sentence_pairs)
            for i, value in enumerate(ce_scores + [0]):
                result[i]['score'] = round(value, 3)
        return result

    def _template(self, canonical: pa.Table,
                  save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                  replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """"""
        # intent recipie options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # code block
