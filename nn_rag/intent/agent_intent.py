from nn_rag.components.commons import Commons
from nn_rag.intent.abstract_agent_intent import AbstractAgentIntentModel

class AgentIntent(AbstractAgentIntentModel):

    def simple_query(self, query: str, connector_name: str=None, save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
              replace_intent: bool=None, remove_duplicates: bool=None):
        """ takes chunks from a Table and converts them to a pyarrow tensor of embeddings.

         :param query: bool text query to run against the list of tensor embeddings
         :param connector_name: a connector name where the question will be applied
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
        if self._pm.has_connector(connector_name):
            handler = self._pm.get_connector_handler(connector_name)
            return handler.load_canonical(query=query)
        raise ValueError(f"The connector name {connector_name} has been given but no Connect Contract added")

