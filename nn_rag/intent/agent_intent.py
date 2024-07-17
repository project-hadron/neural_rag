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

from sentence_transformers import CrossEncoder

from nn_rag.intent.abstract_agent_intent import AbstractAgentIntentModel

class AgentIntent(AbstractAgentIntentModel):

    def simple_query(self, query: str, connector_name: str=None, limit: int=None, save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
              replace_intent: bool=None, remove_duplicates: bool=None):
        """ takes chunks from a Table and converts them to a pyarrow tensor of embeddings.

         :param query: bool text query to run against the list of tensor embeddings
         :param connector_name: a connector name where the question will be applied
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
        limit = limit if isinstance(limit, int) else 5
        if self._pm.has_connector(connector_name):
            handler = self._pm.get_connector_handler(connector_name)
            return handler.load_canonical(query=query, limit=limit)
        raise ValueError(f"The connector name {connector_name} has been given but no Connect Contract added")

    def reranker__query(self, query: str, connector_name: str=None, bi_limit: int=None, cross_limit: int=None,
                        save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                        replace_intent: bool=None, remove_duplicates: bool=None):
        """ takes chunks from a Table and converts them to a pyarrow tensor of embeddings.

         :param query: bool text query to run against the list of tensor embeddings
         :param connector_name: a connector name where the question will be applied
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
        if self._pm.has_connector(connector_name):
            handler = self._pm.get_connector_handler(connector_name)
            result = handler.load_canonical(query=query, limit=bi_limit)

            # cross encoder
            model = CrossEncoder("cross-encoder/stsb-roberta-base", device='cpu')
            sentence_pairs = [[query, s] for s in result['source']]
            ce_scores = model.predict(sentence_pairs)


        raise ValueError(f"The connector name {connector_name} has been given but no Connect Contract added")

