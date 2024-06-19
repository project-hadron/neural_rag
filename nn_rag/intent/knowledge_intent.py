import inspect
import re
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from sentence_transformers import SentenceTransformer, util
from spacy.lang.en import English
from nn_rag.components.commons import Commons
from nn_rag.intent.abstract_knowledge_intent import AbstractKnowledgeIntentModel


class KnowledgeIntent(AbstractKnowledgeIntentModel):
    """This class represents RAG intent actions whereby data preparation can be done
    """

    def str_filter_on_condition(self, canonical: pa.Table, header: str, condition: list, mask_null: bool=None,
                                save_intent: bool=None, intent_order: int=None,
                                intent_level: [int, str]=None, replace_intent: bool=None,
                                remove_duplicates: bool=None) -> pa.Table:
        """ Takes the column name header from the canonical and applies the condition. Where the condition
        is satisfied within the column, the canonical row is removed.

        The selection is a list of triple tuples in the form: [(comparison, operation, logic)] where comparison
        is the item or column to compare, the operation is what to do when comparing and the logic if you are
        chaining tuples as in the logic to join to the next boolean flags to the current. An example might be:

                [(comparison, operation, logic)]
                [(1, 'greater', 'or'), (-1, 'less', None)]
                [(pa.array(['INACTIVE', 'PENDING']), 'is_in', None)]

        The operator and logic are taken from pyarrow.compute and are:

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

    def str_sentence_removal(self, canonical: pa.Table, indices:list=None, to_header: str=None,
                             save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                             replace_intent: bool=None, remove_duplicates: bool=None):
        """ Taking a canonical of sentences from the text_profiler method or allows the given sentence indices
        to be removed.

        'indices' takes a list of either sentence_num, representing the sentence to be removed, or a tuple of start add stop
        range of sentence numbers. For example [1, 3, (5, 8)] would remove the sentences [1, 3, 5, 6, 7]


        :param canonical: a Table of sentences and stats
        :param indices: (optional) a list of numbers and/or tuples for sentences to be dropped
        :param to_header: (optional) an optional name to call the column
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
        sentences = canonical.to_pylist()
        drop_list = []
        for x in indices:
            if isinstance(x, tuple):
                drop_list += (list(range(x[0], x[1])))
            else:
                drop_list += [x]
        df = pd.DataFrame(sentences)
        df = df.drop(index=drop_list)
        return pa.Table.from_pandas(df)


    def str_pattern_replace(self, canonical: pa.Table, header: str, pattern: str, replacement: str, is_regex: bool=None,
                            max_replacements: int=None, to_header: str=None, save_intent: bool=None,
                            intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                            remove_duplicates: bool=None):
        """ For each string in header, replace non-overlapping substrings that match the given literal pattern
        with the given replacement. If max_replacements is given and not equal to -1, it limits the maximum
        amount replacements per input, counted from the left. Null values emit null.

        If is a regex then RE2 Regular Expression Syntax is used

        :param canonical:
        :param header: The name of the target string column
        :param pattern: Substring pattern to look for inside input values.
        :param replacement: What to replace the pattern with.
        :param is_regex: (optional) if the pattern is a regex. Default False
        :param max_replacements: (optional) The maximum number of strings to replace in each input value.
        :param to_header: (optional) an optional name to call the column
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
        header = self._extract_value(header)
        to_header  = self._extract_value(to_header)
        is_regex = is_regex if isinstance(is_regex, bool) else False
        c = canonical.column(header).combine_chunks()
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
        to_header = to_header if isinstance(to_header, str) else header
        return Commons.table_append(canonical, pa.table([rtn_values], names=[to_header]))

    def text_profiler(self, canonical: pa.Table, header: str=None, embedding_name: str=None, max_char_size: int=None,
                      save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                      replace_intent: bool=None, remove_duplicates: bool=None):
        """ Taking a Table with a text column, returning the profile of that text as a list of sentences with
        accompanying statistics to enable discovery.

        :param canonical: a Table with a text column
        :param header: (optional) The name of the target text column, default 'text'
        :param embedding_name: (optional) the name of the embedding model to use to score familiarity
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
        header = self._extract_value(header)
        header = header if isinstance(header, str) else 'text'
        max_char_size = max_char_size if isinstance(max_char_size, int) else 900_000
        embedding_name = self._extract_value(embedding_name)
        embedding_model = SentenceTransformer(model_name_or_path=embedding_name) if embedding_name else None # 'all-mpnet-base-v2'
        nlp = English()
        nlp.add_pipe("sentencizer")
        text = canonical.column(header).to_pylist()
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
            sentences.append({'sentence': s,
                              'sentence_score': 0,
                              'sentence_num': num,
                              "char_count": len(s),
                              "word_count": len(s.split(" ")),
                              "token_count": round(len(s) / 4),  # 1 token = ~4 chars, see:
                              })
            if embedding_name and num < len(sents)-1:
                v1 = embedding_model.encode(s)
                v2 = embedding_model.encode(sents[num+1])
                sentences[num]['sentence_score'] = util.dot_score(v1, v2)[0, 0].tolist()
        return pa.Table.from_pylist(sentences)

    def text_chunker(self, canonical: pa.Table, char_chunk_size: int=None, temperature: float=None,
                     overlap: int=None, save_intent: bool=None, intent_level: [int, str]=None,
                     intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ Taking a profile Table and converts the sentences into chunks ready for embedding. By default,
        the sentences are joined and then chunked according to the chunk_size. However, if the temperature is used
        the sentences are grouped by temperature and then chunked. Be aware you may get small chunks for
        small sentences.

        :param canonical: a text profile Table
        :param char_chunk_size: (optional) The number of characters per chunk. Default is 500
        :param temperature: (optional) a value between 0 and 1 representing the temperature between sentences
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
        char_chunk_size = char_chunk_size if isinstance(char_chunk_size, int) else 256
        overlap = self._extract_value(overlap)
        overlap = overlap if isinstance(overlap, int) else int(char_chunk_size / 10)
        temperature = self._extract_value(temperature)
        temperature = temperature if isinstance(temperature, float) else 1
        nlp = English()
        nlp.add_pipe("sentencizer")
        sentences = canonical.to_pylist()
        if 1 > temperature > 0:
            # sort by score
            parsed_sentence = [sentences[0]["sentence"]]
            score = sentences[0]["sentence_score"]
            for item in sentences[1:]:
                if score > temperature:
                    if len(parsed_sentence[-1]) + item['char_count'] > char_chunk_size:
                        previous = parsed_sentence.pop()
                        *first, last = previous.split('. ')
                        first = [item + '.' for item in first]
                        parsed_sentence.append(first)
                        parsed_sentence.append(last + " " + item["sentence"])
                    else:
                        parsed_sentence[-1] += " " + item["sentence"]
                elif len(parsed_sentence[-1]) + item['char_count'] < char_chunk_size:
                    if score <= temperature:
                        parsed_sentence[-1] += " " + item["sentence"]
                else:
                    parsed_sentence.append(item["sentence"])
                score = item['sentence_score']
        else:
            # text
            parsed_sentence = ''
            for item in sentences:
                parsed_sentence += " " + item['sentence']
            parsed_sentence = [parsed_sentence.strip()]
        # chunks
        chunks = []
        for sentence in parsed_sentence:
            while len(sentence) > 0:
                sentence_chunk = sentence[:char_chunk_size + overlap]
                sentence = sentence[char_chunk_size:]
                chunk_dict = {}
                # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)  # ".A" -> ". A" for any full-stop/capital letter combo
                chunk_dict["chunk_text"] = joined_sentence_chunk

                # Get stats about the chunk
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters

                chunks.append(chunk_dict)
        return pa.Table.from_pylist(chunks)
