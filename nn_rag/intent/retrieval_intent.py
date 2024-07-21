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
import pyarrow as pa
import pyarrow.compute as pc
from sentence_transformers import CrossEncoder

from nn_rag.components.commons import Commons
from nn_rag.intent.abstract_retrieval_intent import AbstractRetrievalIntentModel
from torch import cuda, backends


class RetrievalIntent(AbstractRetrievalIntentModel):

    CONNECTOR_SOURCE = 'primary_source'

    def query_similarity(self, query: str, limit: int=None, save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                         replace_intent: bool=None, remove_duplicates: bool=None):
        """

         :param query: text query to run against the list of tensor embeddings
         :param limit: (optional) the number of items to return from the results of the query
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
        limit = limit if isinstance(limit, int) else 10
        if self._pm.has_connector(self.CONNECTOR_SOURCE):
            handler = self._pm.get_connector_handler(self.CONNECTOR_SOURCE)
            return handler.load_canonical(query=query, limit=limit)
        raise ValueError(f"The source connector has been set")

    def query_reranker(self, query: str, bi_limit: int=None, cross_limit: int=None,
                       save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                       replace_intent: bool=None, remove_duplicates: bool=None):
        """

         :param query: bool text query to run against the list of tensor embeddings
         :param bi_limit: (optional) bi encoder limits for results of the query
         :param cross_limit: (optional) cross encoder limits for results of the query
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
        bi_limit = bi_limit if isinstance(bi_limit, int) else 20
        cross_limit = cross_limit if isinstance(cross_limit, int) else 5
        device = "cuda" if cuda.is_available() else "mps" if hasattr(backends, "mps") and backends.mps.is_available() else "cpu"
        bi_answer = self.query_similarity(query=query, limit=bi_limit)
        # cross encoder
        model = CrossEncoder("cross-encoder/stsb-roberta-base", device=device)
        sentence_pairs = [[query, s] for s in bi_answer['source'].to_pylist()]
        ce_scores = model.predict(sentence_pairs)
        tbl = Commons.table_append(bi_answer, pa.table([pa.array(ce_scores, pa.float32())], names=["cross-encoder_score"]))
        sorted_indices = pc.sort_indices(tbl, sort_keys=[("cross-encoder_score", "descending")])
        return tbl.take(sorted_indices).slice(0, 5)
