import inspect
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from spacy.lang.en import English
from sentence_transformers import util, SentenceTransformer
from nn_rag.components.commons import Commons
from nn_rag.intent.abstract_knowledge_intent import AbstractKnowledgeIntentModel


class KnowledgeIntent(AbstractKnowledgeIntentModel):
    """This class represents RAG intent actions whereby data preparation can be done
    """

    def filter_on_condition(self, canonical: pa.Table, header: str, condition: list, mask_null: bool=None,
                            seed: int=None, save_intent: bool=None, intent_order: int=None,
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
        :param seed: (optional) the random seed. defaults to current datetime
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
        seed = seed if isinstance(seed, int) else self._seed()
        h_col = canonical.column(header).combine_chunks()
        mask = self._extract_mask(h_col, condition=condition, mask_null=mask_null)
        return canonical.filter(mask)

    def sentence_removal(self, canonical: pa.Table, indices:list=None, seed: int=None, to_header: str=None,
                         save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                         replace_intent: bool=None, remove_duplicates: bool=None):
        """ Taking a canonical of sentences from the text_profiler method or allows the given sentence indices
        to be removed.

        'indices' takes a list of either sentence_num, representing the sentence to be removed, or a tuple of start add stop
        range of sentence numbers. For example [1, 3, (5, 8)] would remove the sentences [1, 3, 5, 6, 7]


        :param canonical: a Table of sentences and stats
        :param indices: (optional) a list of numbers and/or tuples for sentences to be dropped
        :param to_header: (optional) an optional name to call the column
        :param seed: (optional) a seed value for the random function: default to None
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
        _seed = seed if isinstance(seed, int) else self._seed()
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


    def pattern_replace(self, canonical: pa.Table, header: str, pattern: str, replacement: str, is_regex: bool=None,
                        max_replacements: int=None, seed: int=None, to_header: str=None, save_intent: bool=None,
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
        :param seed: (optional) a seed value for the random function: default to None
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
        _seed = seed if isinstance(seed, int) else self._seed()
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

    def text_profiler(self, canonical: pa.Table, header: str=None, seed: int=None,
                      save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                      replace_intent: bool=None, remove_duplicates: bool=None):
        """ Taking a Table with a text column, returning the profile of that text as a list of sentences with
        accompanying statistics to enable discovery.

        :param canonical: a Table with a text column
        :param header: (optional) The name of the target text column, default 'text'
        :param seed: (optional) a seed value for the random function: default to None
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
        _seed = seed if isinstance(seed, int) else self._seed()
        nlp = English()
        nlp.add_pipe("sentencizer")
        text = canonical.to_pylist()
        sentences = []
        for item in text:
            sents = list(nlp(item[header]).sents)
            sents = [str(sentence) for sentence in sents]
            for num, s in enumerate(sents):
                sentences.append({'sentence': s,
                                  'sentence_num': num,
                                  "char_count": len(s),
                                  "word_count": len(s.split(" ")),
                                  "token_count": round(len(s) / 4),  # 1 token = ~4 chars, see:
                                  })
        return pa.Table.from_pylist(sentences)

    def sentence_chunks(self, canonical: pa.Table, num_sentence_chunk_size: int=None, seed: int=None,
                        save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                        replace_intent: bool=None, remove_duplicates: bool=None):
        """ Taking a profile Table and converts the sentences into chunks ready for embedding.

        :param canonical: a text profile Table
        :param num_sentence_chunk_size: (optional) The number of sentences in each chunk. Default is 10
        :param seed: (optional) a seed value for the random function: default to None
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
        num_sentence_chunk_size = self._extract_value(num_sentence_chunk_size)
        num_sentence_chunk_size = num_sentence_chunk_size if isinstance(num_sentence_chunk_size, int) else 10
        _seed = seed if isinstance(seed, int) else self._seed()
        nlp = English()
        nlp.add_pipe("sentencizer")
        sentences = canonical.to_pylist()
        chunks = []
        for count, idx in enumerate(range(0, len(sentences), num_sentence_chunk_size)):
            paragraph = ''
            for chunk in sentences[idx:idx + num_sentence_chunk_size]:
                paragraph += chunk['sentence'] + ' '
            paragraph = paragraph.strip()
            chunk_dict = {"chunk_number": count,
                          "chunk_sentence_count": num_sentence_chunk_size, "chunk_char_count": len(paragraph),
                          "chunk_word_count": len([word for word in paragraph.split(" ")]),
                          "chunk_token_count": round(len(paragraph) / 4),
                          "chunk_text": paragraph.strip(),}
            chunks.append(chunk_dict)
        return pa.Table.from_pylist(chunks)

    def chunk_embedding(self, canonical: pa.Table, batch_size: int=None, embedding_name: str=None, device: str=None,
                        seed: int=None, save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                        replace_intent: bool=None, remove_duplicates: bool=None):
        """ takes chunks from a Table and converts them to a pyarrow tensor of embeddings.

         :param canonical: sentence chunks to be embedded
         :param batch_size: (optional) the size of the embedding batches
         :param embedding_name: (optional) the name of the embedding algorithm to use with sentence_transformer
         :param device: (optional) the device types to use for example 'cpu', 'gpu', 'cuda'
         :param seed: (optional) a seed value for the random function: default to None
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
        _seed = seed if isinstance(seed, int) else self._seed()
        batch_size = self._extract_value(batch_size)
        batch_size = batch_size if isinstance(batch_size, int) else 32
        embedding_name = self._extract_value(embedding_name)
        embedding_name = embedding_name if isinstance(embedding_name, str) else 'all-mpnet-docker-v2'
        device = self._extract_value(device)
        device = device if isinstance(device, str) else 'cpu'
        chunks = canonical.to_pylist()
        embedding_model = SentenceTransformer(model_name_or_path=embedding_name, device=device)
        # Turn text chunks into a single list
        text_chunks = [item["chunk_text"] for item in chunks]
        numpy_embedding = embedding_model.encode(text_chunks, batch_size=batch_size, convert_to_numpy=True)
        return pa.Tensor.from_numpy(numpy_embedding)

    def score_embedding(self, canonical: pa.Tensor, query: str, topk: int=None, embedding_name: str=None, device: str=None,
                        seed: int=None, save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                        replace_intent: bool=None, remove_duplicates: bool=None):
        """ takes chunks from a Table and converts them to a pyarrow tensor of embeddings.

         :param canonical: a list of py-arrow tensors
         :param query: bool text query to run against the list of tensor embeddings
         :param topk: (optional) the top k number of embeddings that fit the query
         :param embedding_name: (optional) the name of the embedding algorithm to use with sentence_transformer
         :param device: (optional) the device types to use for example 'cpu', 'gpu', 'cuda'
         :param seed: (optional) a seed value for the random function: default to None
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
        query = self._extract_value(query)
        _seed = seed if isinstance(seed, int) else self._seed()
        topk = self._extract_value(topk)
        topk = topk if isinstance(topk, int) else 5
        embedding_name = self._extract_value(embedding_name)
        embedding_name = embedding_name if isinstance(embedding_name, str) else 'all-mpnet-docker-v2'
        device = self._extract_value(device)
        device = device if isinstance(device, str) else 'cpu'
        embedding_model = SentenceTransformer(model_name_or_path=embedding_name, device=device)
        query_embedding = embedding_model.encode(query, convert_to_tensor=True)
        embeddings = torch.tensor(canonical.to_numpy(), dtype=torch.float32, device='cpu')
        dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
        scores, indices = torch.topk(input=dot_scores, k=topk)
        return scores, indices
