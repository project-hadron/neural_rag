import inspect

from nn_rag import Knowledge
from nn_rag.components.commons import Commons
from ds_core.intent.abstract_intent import AbstractIntentModel
from nn_rag.managers.controller_property_manager import ControllerPropertyManager


class ControllerIntentModel(AbstractIntentModel):

    """This components provides a set of actions that focuses on the Controller. The Controller is a unique components
    that independently orchestrates the components registered to it. It executes the components Domain Contract and
    not its code. The Controller orchestrates how those components should run with the components being independent
    in their actions and therefore a separation of concerns."""

    def __init__(self, property_manager: ControllerPropertyManager, default_save_intent: bool=None,
                 default_intent_level: [str, int, float]=None, order_next_available: bool=None,
                 default_replace_intent: bool=None):
        """initialisation of the Intent class.

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param default_intent_level: (optional) the default level intent should be saved at
        :param order_next_available: (optional) if the default behaviour for the order should be next available order
        :param default_replace_intent: (optional) the default replace existing intent behaviour
        """
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_replace_intent = default_replace_intent if isinstance(default_replace_intent, bool) else True
        default_intent_level = default_intent_level if isinstance(default_intent_level, (str, int, float)) else 'docker'
        default_intent_order = -1 if isinstance(order_next_available, bool) and order_next_available else 0
        intent_param_exclude = ['canonical']
        intent_type_additions = []
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         intent_param_exclude=intent_param_exclude, default_intent_level=default_intent_level,
                         default_intent_order=default_intent_order, default_replace_intent=default_replace_intent,
                         intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, run_level: str, source: str=None, persist: [str, list]=None,
                            controller_repo: str=None, **kwargs):
        """ Collectively runs all parameterised intent taken from the property manager against the code docker as
        defined by the intent_contract.

        It is expected that all intent methods have the 'canonical' as the first parameter of the method signature
        and will contain 'save_intent' as parameters.

        :param run_level:
        :param persist:
        :param source:
        :param controller_repo: (optional) the controller repo to use if no uri_pm_repo is within the intent parameters
        :param kwargs: additional kwargs to add to the parameterised intent, these will replace any that already exist
        :return: Canonical with parameterised intent applied
        """
        # get the list of levels to run
        if not self._pm.has_intent(run_level):
            raise ValueError(f"The intent level '{run_level}' could not be found in the "
                             f"property manager '{self._pm.manager_name()}' for task '{self._pm.task_name}'")
        shape = None
        level_key = self._pm.join(self._pm.KEY.intent_key, run_level)
        for order in sorted(self._pm.get(level_key, {})):
            for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                if method in self.__dir__():
                    # failsafe in case kwargs was stored as the reference
                    params.update(params.pop('kwargs', {}))
                    # add method kwargs to the params
                    if isinstance(kwargs, dict):
                        params.update(kwargs)
                    # remove the creator param
                    _ = params.pop('intent_creator', 'Unknown')
                    # add excluded params and set to False
                    params.update({'save_intent': False})
                    # run the action
                    params.update({'register_only': False})
                    # add the controller_repo if given
                    if isinstance(controller_repo, str) and 'uri_pm_repo' not in params.keys():
                        params.update({'uri_pm_repo': controller_repo})
                    shape = eval(f"self.{method}(source=source, persist=persist, **{params})", globals(), locals())
        return shape

    def knowledge(self, task_name: str, source: str=None, persist: [str, list]=None, columns: [str, list]=None,
                  register_only: bool=None, save_intent: bool=None, intent_order: int=None, intent_level: [int, str]=None,
                  replace_intent: bool=None, remove_duplicates: bool=None, **kwargs):
        """ register a Knowledge component task pipeline

        :param persist:
        :param source:
        :param task_name: the task_name reference for this component
        :param columns: (optional) a single or list of intent_level to run, if list, run in order given
        :param register_only: (optional) if the action should only be registered and not run
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
       """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # create the
        register_only = register_only if isinstance(register_only, bool) else True
        if register_only:
            return
        kn: Knowledge = eval(f"Knowledge.from_env(task_name=task_name, default_save=False, "
                                f"has_contract=True, **{kwargs})", globals(), locals())
        if source and kn.pm.has_connector(source):
            canonical = kn.load_canonical(source)
        elif kn.pm.has_connector(kn.CONNECTOR_SOURCE):
            canonical = kn.load_source_canonical()
        else:
            canonical = None
        canonical = kn.intent_model.run_intent_pipeline(canonical=canonical, intent_levels=intent_level)
        if persist:
            for out in Commons.list_formatter(persist):
                if kn.pm.has_connector(out):
                    kn.save_canonical(connector_name=out, canonical=canonical)
        else:
            kn.save_persist_canonical(canonical=canonical)
        return

