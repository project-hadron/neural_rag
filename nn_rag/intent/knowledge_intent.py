import ast
import inspect
import re
from collections import Counter
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import spacy
from spacy.language import Language
from sentence_transformers import SentenceTransformer, util
from nn_rag.components.commons import Commons
from nn_rag.intent.abstract_knowledge_intent import AbstractKnowledgeIntentModel


class KnowledgeIntent(AbstractKnowledgeIntentModel):
    """This class represents RAG intent actions whereby data preparation can be done
    """

    def filter_on_condition(self, canonical: pa.Table, header: str, condition: list, mask_null: bool=None,
                            save_intent: bool=None, intent_order: int=None, intent_level: [int, str]=None,
                            replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
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
        # reset the index
        return canonical.drop('index').add_column(0, 'index', [list(range(canonical.num_rows))])

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
        :param indices: (optional) a list of numbers and/or tuples for rows to be dropped
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
            canonical = pa.table(filtered_dict)
            # reset the index
            return canonical.drop('index').add_column(0, 'index', [list(range(canonical.num_rows))])
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
        if is_regex:
            rtn_values = pc.replace_substring_regex(c, pattern, replacement, max_replacements=max_replacements)
        else:
            rtn_values = pc.replace_substring(c, pattern, replacement, max_replacements=max_replacements)
        canonical = Commons.table_append(canonical, pa.table([rtn_values], names=['text']))
        # reset the index
        return canonical.drop('index').add_column(0, 'index', [list(range(canonical.num_rows))])

    def filter_on_join(self, canonical: pa.Table, indices: list, save_intent: bool=None, intent_level: [int, str]=None,
                       intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ Takes a list of indices and joins those indices with the next row as a sum of the two. This allows
        two rows with high similarity scores to be joined together.

        :param canonical: a pa.Table as the reference table
        :param indices: (optional) a list of index values to be joined to the following row
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
        indices = Commons.list_formatter(indices)
        df = pd.DataFrame(canonical.to_pandas())
        for idx in range(1, df.shape[0]):
            if idx - 1 in indices:
                df.loc[idx, 'index'] = df.loc[idx - 1, 'index']
        df = df.groupby('index').sum()
        df['text'] = df['text'].apply(lambda x: re.sub(r'\.([A-Z])', r' \1', x))
        canonical = pa.Table.from_pandas(df)
        # reset the index
        return canonical.drop('index').add_column(0, 'index', [list(range(canonical.num_rows))])

    def text_to_document(self, canonical: pa.Table, sep: str=None, save_intent: bool=None,
                         intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                         remove_duplicates: bool=None) -> pa.Table:
        """ Takes a table and joins all the row text into a single row.

        :param canonical: a pa.Table as the reference table
        :param sep: (optional) bin seperator between joining chunks. The default is '\n'
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
        total = sep.join(text)
        full_text = [
            {"index": 1,
             "text_char_count": len(text),
             "text_token_count": round(len(text) / 4),
             "text": total}
        ]
        return pa.Table.from_pylist(full_text)

    def text_to_paragraphs(self, canonical: pa.Table, include_score: bool=None, sep: str=None, words_max: int=None,
                           words_threshold: int=None, words_type: list=None, max_char_size: int=None,
                           save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                           replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ Takes a table with the text column and split it into perceived paragraphs. This method
        is generally used for text discovery and manipulation before chunking.

        :param canonical: a pa.Table as the reference table
        :param include_score: (optional) if the score should be calculated. This helps with speed. Default is True
        :param words_max: (optional) the maximum number of words to display and score. Default is 8
        :param words_threshold: (optional) the threshold count of repeating words. Default is 2
        :param words_type: (optional) a list of word types eg. ['NOUN','PROPN','VERB','ADJ'], Default['NOUN','PROPN']
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
        include_score = include_score if isinstance(include_score, bool) else True
        words_max = words_max if isinstance(words_max, int) else 8
        words_threshold = words_threshold if isinstance(words_threshold, int) else 2
        words_type = words_type if isinstance(words_type, list) else ['NOUN','PROPN']
        sep = self._extract_value(sep)
        sep = sep if isinstance(sep, str) else '\n\n'
        max_char_size = max_char_size if isinstance(max_char_size, int) else 900_000
        # load English parser
        text = canonical.column('text').to_pylist()
        chunked_text = []
        for item in text:
            chunked_text += [item[i:i + max_char_size] for i in range(0, len(item), max_char_size)]
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("custom_sentencizer", before="parser")
        text = [x.replace(sep, ' | ') for x in chunked_text]
        sep_para = []
        for item in text:
            doc = nlp(item)
            for sent in doc.sents:
                sep_para.append(str(sent.text).replace(' |', ' ').replace('\n', ' ').strip())
        paragraphs = []
        for num, p in enumerate(sep_para):
            if words_max > 0:
                doc = nlp(p)
                words = [token.text for token in doc if token.pos_ in words_type]
                common_words = Counter(words).most_common(words_max)
                words_freq = [k.lower() for k, c in common_words if c >= words_threshold]
            else:
                words_freq = []
            paragraphs.append({
                "index": num,
                "paragraph_char_count": len(p),
                "paragraph_word_count": len(p.split(" ")),
                "paragraph_sentence_count": len(p.split(". ")),
                "paragraph_token_count": round(len(p) / 4),  # 1 token = ~4 chars, see:
                "paragraph_score": 0,
                "paragraph_words": words_freq,
                'text': p,
            })
        if include_score:
            # set embedding
            embedding_model = SentenceTransformer(model_name_or_path='all-mpnet-base-v2')
            for num, item in enumerate(paragraphs):
                if num >= len(paragraphs) -1:
                    break
                if not item['paragraph_words'] or not paragraphs[num+1]['paragraph_words']:
                    continue
                v1 = embedding_model.encode(' '.join(item['paragraph_words']))
                v2 = embedding_model.encode(' '.join(paragraphs[num+1]['paragraph_words']))
                paragraphs[num]['paragraph_score'] = round(util.dot_score(v1, v2)[0, 0].tolist(), 3)
        return pa.Table.from_pylist(paragraphs)

    def text_to_sentences(self, canonical: pa.Table, include_score: bool=None, max_char_size: int=None, words_max: int=None,
                          words_threshold: int=None, words_type: int=None, save_intent: bool=None, intent_level: [int, str]=None,
                          intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ Taking a Table with a text column, returning the profile of that text as a list of sentences. This method
        is generally used for text discovery and manipulation before chunking.

        :param canonical: a pa.Table as the reference table
        :param include_score: (optional) if the score should be included. This helps with speed. Default is True
        :param max_char_size: (optional) the maximum number of characters to process at one time
        :param words_max: (optional) the maximum number of words to display and score. Default is 5
        :param words_threshold: (optional) the threshold count of repeating words. Default is 1
        :param words_type: (optional) a list of word types eg. ['NOUN','PROPN','VERB','ADJ'], Default ['NOUN','PROPN']
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
        include_score = include_score if isinstance(include_score, bool) else True
        max_char_size = max_char_size if isinstance(max_char_size, int) else 900_000
        words_max = words_max if isinstance(words_max, int) else 5
        words_threshold = words_threshold if isinstance(words_threshold, int) else 1
        words_type = words_type if isinstance(words_type, list) else ['NOUN','PROPN']
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
            if words_max > 0:
                doc = nlp(s)
                words = [token.text for token in doc if token.pos_ in words_type]
                common_words = Counter(words).most_common(words_max)
                words_freq = [k.lower() for k, c in common_words if c >= words_threshold]
            else:
                words_freq = []
            sentences.append({
                "index": num,
                "sentence_char_count": len(s),
                "sentence_word_count": len(s.split(" ")),
                "sentence_token_count": round(len(s) / 4),  # 1 token = ~4 chars, see:
                'sentence_score': 0,
                "sentence_words": words_freq,
                'text': s,
            })
        if include_score:
            # set embedding
            embedding_model = SentenceTransformer(model_name_or_path='all-mpnet-base-v2')
            for num, item in enumerate(sentences):
                if num >= len(sentences) -1:
                    break
                if not item['sentence_words'] or not sentences[num+1]['sentence_words']:
                    continue
                v1 = embedding_model.encode(' '.join(item['sentence_words']))
                v2 = embedding_model.encode(' '.join(sentences[num+1]['sentence_words']))
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
