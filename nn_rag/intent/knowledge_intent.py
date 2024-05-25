import inspect
import re
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from nn_rag.components.commons import Commons
from nn_rag.intent.abstract_knowledge_intent import AbstractKnowledgeIntentModel


class KnowledgeIntent(AbstractKnowledgeIntentModel):
    """This class represents RAG intent actions whereby data preparation can be done
    """

    def correlate_on_pandas(self, canonical: pa.Table, header: str, code_str: str, to_header: str=None, seed: int=None,
                            save_intent: bool=None, intent_order: int=None, intent_level: [int, str]=None,
                            replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """ Allows a Pandas Series method to be run against a Table column. Examples of code_str:
                "str.extract('([0-9]+)').astype('float')"
                "apply(lambda x: x[0] if isinstance(x, str) else None)"

        :param canonical: a pa.Table as the reference table
        :param header: the header for the target values to change
        :param code_str: a code string matching a Pandas Series method such as str or apply
        :param to_header: (optional) an optional name to call the column
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
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        canonical = self._get_canonical(canonical)
        header = self._extract_value(header)
        to_header  = self._extract_value(to_header)
        if not isinstance(header, str) or header not in canonical.column_names:
            raise ValueError(f"The header '{header}' can't be found in the canonical headers")
        seed = seed if isinstance(seed, int) else self._seed()
        s_values = canonical.column(header).combine_chunks().to_pandas()
        s_values = eval(f's_values.{code_str}', globals(), locals())
        if isinstance(s_values, pd.DataFrame):
            s_values = s_values.iloc[:, 0]
        rtn_arr = pa.Array.from_pandas(s_values)
        to_header = to_header if isinstance(to_header, str) else header
        return Commons.table_append(canonical, pa.table([rtn_arr], names=[to_header]))

    def correlate_replace(self, canonical: pa.Table, header: str, pattern: str, replacement: str, is_regex: bool=None,
                          max_replacements: int=None, seed: int=None, to_header: str=None, save_intent: bool=None,
                          intent_level: [int, str]=None, intent_order: int=None, replace_intent: bool=None,
                          remove_duplicates: bool=None):
        """ For each string in target, replace non-overlapping substrings that match the given literal pattern
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

    def text_profiler(self, canonical: pa.Table, as_text: bool=None, header: str=None, num_sentence_chunk_size: int=None,
                      to_drop:list=None, seed: int=None, save_intent: bool=None, intent_level: [int, str]=None,
                      intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ Taking a Table with a text column, returning the profile of that text as a list of sentences
        that can be interrogated and the selective sentence numbers removed. By default this returns the text
        profile as a set of sentences with accompanying statistics to enable discovery. It can also be set to return
        a text column where the to_drop list has been applied, and thus part of a pipeline.

        to_drop takes a list of either integers, representing the sentence to be removed, or a tuple of start add stop
        range of sentence numbers. For example [1, 3, (5, 8)] would remove the sentences [1, 3, 5, 6, 7]

        :param canonical: a Table with a text column
        :param as_text: (optional) If this should return a text column or a full profile for discovered.
        :param header: (optional) The name of the target text column, default 'text'
        :param num_sentence_chunk_size: (optional) the number of sentences to chunk, default is 10
        :param to_drop: (optional) a list of numbers and/or tuples for sentences to be dropped
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
        num_sentence_chunk_size = self._extract_value(num_sentence_chunk_size)
        num_sentence_chunk_size = num_sentence_chunk_size if isinstance(num_sentence_chunk_size, int) else 10

        to_drop = Commons.list_formatter(to_drop)
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
                                  "token_count": len(s) / 4,  # 1 token = ~4 chars, see:
                                  })
        if len(to_drop) > 0:
            drop_list = []
            for x in to_drop:
                if isinstance(x, tuple):
                    drop_list += (list(range(x[0], x[1])))
                else:
                    drop_list += [x]
            drop_list.sort()
            df = pd.DataFrame(sentences)
            df = df.drop(index=drop_list)
            sentences = df.to_dict(orient='records')
        if as_text:
            text = ''
            for idx in range(len(sentences)):
                text += sentences[idx]['sentence'] + ' '
            arr = pa.array([text.strip()], pa.string())
            return pa.table([arr], names=['text'])
        return pa.Table.from_pylist(sentences)


    def text_chunk(self, canonical: pa.Table, text_name: str, header: str=None, num_sentence_chunk_size: int=None,
                   seed: int=None, save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                   replace_intent: bool=None, remove_duplicates: bool=None):
        """ Taking a Table with a text column, converts the text into chunks ready for embedding.

        to_drop takes a list of either integers, representing the sentence to be removed, or a tuple of start and stop
        range of sentence numbers. For example [1, 3, (5, 8)] would remove the sentences [1, 3, 5, 6, 7]. These values
        can be identified from the text_profiling as part of discovery.

        :param canonical: a text profile Table
        :param text_name: a reference name for this text
        :param header: (optional) The name of the text column in the Table. Default is 'text'
        :param num_sentence_chunk_size: (optional) The number of sentences in each chunk. Default is 10
        :param to_drop: (optional) a list of numbers and/or tuples for sentences to be dropped
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
        header = header if isinstance(header, str) else 'text'
        num_sentence_chunk_size = num_sentence_chunk_size if isinstance(num_sentence_chunk_size, int) else 10
        _seed = seed if isinstance(seed, int) else self._seed()
        nlp = English()
        nlp.add_pipe("sentencizer")
        text = canonical.to_pylist()
        pages_and_chunks = []
        for item in text:
            item["sentences"] = list(nlp(item[header]).sents)
            # Make sure all sentences are strings
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
            item["sentence_chunks"] = [item["sentences"][i:i + num_sentence_chunk_size] for i in
                                       range(0, len(item["sentences"]), num_sentence_chunk_size)]
            for count, sentence_chunk in enumerate(item["sentence_chunks"]):
                chunk_dict = {"text_name": text_name, "chunk_number": count}
                # Join the sentences together into a paragraph-like structure, aka join the list of sentences into one paragraph
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1',
                                               joined_sentence_chunk)  # ".A" => ". A" (will work for any captial letter)
                chunk_dict["sentence_chunks"] = joined_sentence_chunk
                # Get some stats on our chunks
                chunk_dict["chunk_sentence_count"] = num_sentence_chunk_size
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 chars

                pages_and_chunks.append(chunk_dict)
        return pa.Table.from_pylist(pages_and_chunks)

    def chunk_embedding(self, canonical: pa.Table, batch_size: int=None, embedding_name: str=None, device: str=None,
                        seed: int=None, save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                        replace_intent: bool=None, remove_duplicates: bool=None):
        """ takes sentence chunks from a Table and converts them to a pyarrow tensor.

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
        pages_and_chunks = canonical.to_pylist()
        embedding_model = SentenceTransformer(model_name_or_path=embedding_name, device=device)
        # Turn text chunks into a single list
        text_chunks = [item["sentence_chunk"] for item in pages_and_chunks]
        numpy_embedding = embedding_model.encode(text_chunks, batch_size=batch_size, convert_to_numpy=True)
        return pa.Tensor.from_numpy(numpy_embedding)

