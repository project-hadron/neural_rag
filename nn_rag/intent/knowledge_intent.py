import ast
import inspect
import re
from collections import Counter
import pyarrow as pa
import pyarrow.compute as pc
import spacy
from sentence_transformers import SentenceTransformer, util
from spacy.language import Language
from nn_rag.components.commons import Commons
from nn_rag.intent.abstract_knowledge_intent import AbstractKnowledgeIntentModel


class KnowledgeIntent(AbstractKnowledgeIntentModel):
    """This class represents RAG intent actions whereby data preparation can be done
    """

    def filter_on_condition(self, canonical: pa.Table, header: str, condition: list, mask_null: bool=None,
                            save_intent: bool=None, intent_order: int=None, intent_level: [int, str]=None,
                            replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ Takes the column name header from the canonical and applies the condition. Where the condition
        is satisfied within the column, the canonical row is removed.

        The selection is a list of triple tuples in the form: [(comparison, operation, logic)] where comparison
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
        return canonical.filter(mask)

    def filter_on_mask(self, canonical: pa.Table, indices: list=None, pattern: str=None, save_intent: bool=None,
                       intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                       remove_duplicates: bool=None) -> pa.Table:
        """ Taking a canonical with a text column and removes based on either a regex
        pattern or list of index.

        'indices' takes a list of index to be removed, or/and tuples of start add stop
        range of index numbers. For example [1, 3, (5, 8)] would remove the index
        [1, 3, 5, 6, 7].

        'pattern' takes a regex str to find within the text from which that row is
        removed. For example '^Do Not Use Without Permission' would remove rows
        where the text starts with that string.

        :param canonical: a pa.Table as the reference table
        :param indices: (optional) a list of numbers and/or tuples for sentences to be dropped
        :param pattern: (optional) a regex expression pattern to remove an element
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
            # Convert the table to a list of rows
            table_as_list = canonical.to_pydict()
            # Create a new dictionary excluding the specified indices
            filtered_dict = {key: [value for i, value in enumerate(column) if i not in indices]
                             for key, column in table_as_list.items()}
            # Convert the filtered dictionary back to an Arrow table
            tbl = pa.table(filtered_dict)
            # reset the index
            t2 = pa.table([pa.array([int(x) for x in range(tbl.num_rows)])], names=['index'])
            return Commons.table_append(tbl, t2)
        return canonical

    def filter_replace_str(self, canonical: pa.Table, pattern: str, replacement: str, is_regex: bool=None,
                           max_replacements: int=None, save_intent: bool=None, intent_level: [int, str]=None,
                           intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ For each string in 'text', replace non-overlapping substrings that match the given literal pattern
        with the given replacement. If max_replacements is given and not equal to -1, it limits the maximum
        amount replacements per input, counted from the left. Null values emit null.

        If is a regex then RE2 Regular Expression Syntax is used

        :param canonical: a pa.Table as the reference table
        :param pattern: Substring pattern to look for inside input values.
        :param replacement: What to replace the pattern with.
        :param is_regex: (optional) if the pattern is a regex. Default False
        :param max_replacements: (optional) The maximum number of strings to replace in each input value.
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
        is_regex = is_regex if isinstance(is_regex, bool) else False
        c = canonical.column('text').combine_chunks()
        is_dict = False
        if pa.types.is_dictionary(c.type):
            is_dict = True
            c = c.dictionary_decode()
        if is_regex:
            rtn_values = pc.replace_substring_regex(c, pattern, replacement, max_replacements=max_replacements)
        else:
            rtn_values = pc.replace_substring(c, pattern, replacement, max_replacements=max_replacements)
        if is_dict:
            rtn_values = rtn_values.dictionary_encode()
        return Commons.table_append(canonical, pa.table([rtn_values], names=['text']))

    def text_join(self, canonical: pa.Table, save_intent: bool=None, intent_level: [int, str]=None,
                  intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ Takes a table and joins all the row text into a single row.

        :param canonical: a pa.Table as the reference table
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
        text = canonical.column('text').to_pylist()
        total = ''
        for item in text:
            total = ' '.join(item)
        t_array = pa.array([str(total)], pa.string())
        i_array = pa.array([int(x) for x in range(len(t_array))])
        return pa.table([i_array, t_array], names=['index', 'text'])

    def text_to_paragraphs(self, canonical: pa.Table, top_words: int=None, threshold_words: int=None,
                           top_nouns: int=None, threshold_nouns: int=None, sep: str=None,
                           max_char_size: int=None, save_intent: bool=None, intent_level: [int, str]=None,
                           intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ Takes a table with the text column and split it into perceived paragraphs. This method
        is generally used for text discovery and manipulation before chunking.

        :param canonical: a pa.Table as the reference table
        :param top_words: (optional) the minimum number of repeated words to show
        :param threshold_words: (optional)
        :param top_nouns: (optional)
        :param threshold_nouns: (optional)
        :param sep: (optional) The separator patter for the paragraphs
        :param max_char_size: (optional) the maximum number of characters to process at one time
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

        @Language.component("custom_sentencizer")
        def custom_sentencizer(document):
            for i, token in enumerate(document[:-2]):
                # Define sentence start if pipe + titlecase token
                if token.text == "|" and document[i + 1].is_title:
                    document[i + 1].is_sent_start = True
                else:
                    # Explicitly set sentence start to False otherwise, to tell
                    # the parser to leave those tokens alone
                    document[i + 1].is_sent_start = False
            return document

        canonical = self._get_canonical(canonical)
        top_words = top_words if isinstance(top_words, int) else 4
        top_nouns = top_nouns if isinstance(top_nouns, int) else 4
        threshold_words = threshold_words if isinstance(threshold_words, int) else 1
        threshold_nouns = threshold_nouns if isinstance(threshold_nouns, int) else 1
        sep = self._extract_value(sep)
        sep = sep if isinstance(sep, str) else '\n\n'
        max_char_size = max_char_size if isinstance(max_char_size, int) else 900_000
        text = canonical.column('text').to_pylist()
        # load English parser
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("custom_sentencizer", before="parser")
        sub_text = []
        for item in text:
            sub_text += [item[i:i + max_char_size] for i in range(0, len(item), max_char_size)]
        text = [x.replace(sep, ' | ') for x in sub_text]
        sent_para = []
        for item in text:
            doc = nlp(item)
            for sent in doc.sents:
                sent_para.append(str(sent.text).replace(' |', '.').replace('\n', ' ').strip())
        paragraphs = []
        for num, p in enumerate(sent_para):
            words_freq = {}
            doc = nlp(p)
            words = [token.text for token in doc
                     if not token.is_stop and
                     not token.is_punct and
                     not token.is_space and
                     not token.text in ['●']]
            common_words = Counter(words).most_common(top_words)
            words_freq = [k for k, c in common_words if c >= threshold_words]

            nouns = [token.text
                     for token in doc
                     if (not token.is_stop and
                         not token.is_punct and
                         not token.is_space and
                         not token.text in ['●'] and
                         token.pos_ == "NOUN")]
            common_nouns = Counter(nouns).most_common(top_nouns)
            nouns_freq = [k for k, c in common_nouns if c >= threshold_nouns]

            paragraphs.append({
                "index": num,
                "paragraph_char_count": len(p),
                "paragraph_word_count": len(p.split(" ")),
                "paragraph_sentence_count": len(p.split(". ")),
                "paragraph_token_count": round(len(p) / 4),  # 1 token = ~4 chars, see:
                "paragraph_score": 0,
                "paragraph_words": words_freq,
                "paragraph_nouns" : nouns_freq,
                'text': p,
            })
        # set embedding
        embedding_model = SentenceTransformer(model_name_or_path='all-mpnet-base-v2')
        for num, item in enumerate(paragraphs):
            if num >= len(paragraphs) -1:
                break
            if not item['paragraph_nouns']:
                continue
            v1 = embedding_model.encode(' '.join(item['paragraph_nouns']))
            v2 = embedding_model.encode(paragraphs[num+1]['text'])
            paragraphs[num]['paragraph_score'] = round(util.dot_score(v1, v2)[0, 0].tolist(), 3)
        return pa.Table.from_pylist(paragraphs)

    def text_to_sentences(self, canonical: pa.Table, max_char_size: int=None, save_intent: bool=None,
                          intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                          remove_duplicates: bool=None) -> pa.Table:
        """ Taking a Table with a text column, returning the profile of that text as a list of sentences. This method
        is generally used for text discovery and manipulation before chunking.

        :param canonical: a pa.Table as the reference table
        :param max_char_size: (optional) the maximum number of characters to process at one time
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
        max_char_size = max_char_size if isinstance(max_char_size, int) else 900_000
        # SpaCy
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("sentencizer")
        text = canonical.column('text').to_pylist()
        sub_text = []
        for item in text:
            sub_text += [item[i:i + max_char_size] for i in range(0, len(item), max_char_size)]
        text = sub_text
        sents=[]
        for item in text:
            sents += list(nlp(item).sents)
            sents = [str(sentence) for sentence in sents]
        sentences = []
        for num, s in enumerate(sents):
            doc = nlp(s)
            chunks = [token.text for token in doc.noun_chunks]
            sentences.append({
                "index": num,
                "sentence_char_count": len(s),
                "sentence_word_count": len(s.split(" ")),
                "sentence_token_count": round(len(s) / 4),  # 1 token = ~4 chars, see:
                'sentence_score': 0,
                "sentence_noun_chunks": chunks,
                'text': s,
            })
        # set embedding
        embedding_model = SentenceTransformer(model_name_or_path='all-mpnet-base-v2')
        for num, item in enumerate(sentences):
            if num >= len(sentences) -1:
                break
            if not item['text']:
                continue
            v1 = embedding_model.encode(item['text'])
            v2 = embedding_model.encode(sentences[num+1]['text'])
            sentences[num]['sentence_score'] = round(util.dot_score(v1, v2)[0, 0].tolist(), 3)
        return pa.Table.from_pylist(sentences)

    def text_to_chunks(self, canonical: pa.Table, char_chunk_size: int=None, overlap: int=None,
                       save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                       replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ Taking a profile Table and converts the sentences into chunks ready for embedding. By default,
        the sentences are joined and then chunked according to the chunk_size. However, if the temperature is used
        the sentences are grouped by temperature and then chunked. Be aware you may get small chunks for
        small sentences.

        :param canonical: a pa.Table as the reference table
        :param char_chunk_size: (optional) The number of characters per chunk. Default is 500
        :param overlap: (optional) the number of chars a chunk should overlap. Note this adds to the size of the chunk
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
        char_chunk_size = self._extract_value(char_chunk_size)
        char_chunk_size = char_chunk_size if isinstance(char_chunk_size, int) else 500
        overlap = self._extract_value(overlap)
        overlap = overlap if isinstance(overlap, int) else int(char_chunk_size / 10)
        text = canonical.column('text').to_pylist()
        chunks = []
        for item in text:
            while len(item) > 0:
                text_chunk = item[:char_chunk_size + overlap]
                item = item[char_chunk_size:]
                chunk_dict = {}
                # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
                joined_text_chunk = "".join(text_chunk).replace("  ", " ").strip()
                joined_text_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_text_chunk)  # ".A" -> ". A" for any full-stop/capital letter combo
                # Get stats about the chunk
                chunk_dict["chunk_char_count"] = len(joined_text_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_text_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_text_chunk) / 4  # 1 token = ~4 characters
                chunk_dict["text"] = joined_text_chunk
                chunks.append(chunk_dict)
        return pa.Table.from_pylist(chunks)

    #  ---------
    #   Private
    #  ---------

    def _template(self, canonical: pa.Table,
                  save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                  replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """"""
        # intent recipie options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # code block
