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

import pyarrow as pa
from ds_core.intent.abstract_intent import AbstractIntentModel
from ds_core.handlers.abstract_handlers import ConnectorContract
from nn_rag.managers.generator_property_manager import GeneratorPropertyManager

class AbstractGeneratorIntentModel(AbstractIntentModel):

    _INTENT_PARAMS = ['self', 'save_intent', 'intent_level', 'intent_order',
                      'replace_intent', 'remove_duplicates']

    def __init__(self, property_manager: GeneratorPropertyManager, default_save_intent: bool=None,
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
        default_intent_level = default_intent_level if isinstance(default_intent_level, (str, int, float)) else 'primary'
        default_intent_order = -1 if isinstance(order_next_available, bool) and order_next_available else 0
        intent_param_exclude = ['canonical', 'model', 'tokenizer']
        intent_type_additions = []
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         intent_param_exclude=intent_param_exclude, default_intent_level=default_intent_level,
                         default_intent_order=default_intent_order, default_replace_intent=default_replace_intent,
                         intent_type_additions=intent_type_additions)
        self._model = None
        self._tokenizer = None

    @classmethod
    def __dir__(cls):
        """returns the list of available methods associated with the parameterized intent"""
        rtn_list = []
        for m in dir(cls):
            if not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    def run_intent_pipeline(self, canonical: pa.Table=None, intent_level: [str, int]=None,
                            simulate: bool=None, **kwargs) -> pa.Table:
        """Collectively runs all parameterised intent taken from the property manager against the code docker as
        defined by the intent_contract.

        :param canonical: a direct or generated pd.DataFrame. see context notes below
        :param intent_level: (optional) a single intent_level to run
        :param simulate: (optional) returns a report of the order of run and return the indexed column order of run
        :return: a pa.Table
        """
        simulate = simulate if isinstance(simulate, bool) else False
        intent_level = intent_level if isinstance(intent_level, (str, int)) else self._default_intent_level
        col_sim = {"column": [], "order": [], "method": []}
        canonical = self._get_canonical(canonical)
        size = canonical.shape[0] if isinstance(canonical, pa.Table) else kwargs.pop('size', 1000)
        # test if there is any intent to run
        if not self._pm.has_intent(intent_level):
            raise ValueError(f"intent '{intent_level}' is not in [{self._pm.get_intent()}]")
        level_key = self._pm.join(self._pm.KEY.intent_key, intent_level)
        for order in sorted(self._pm.get(level_key, {})):
            for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                try:
                    if method in self.__dir__():
                        if simulate:
                            col_sim['column'].append(intent_level)
                            col_sim['order'].append(order)
                            col_sim['method'].append(method)
                            continue
                        params.update(params.pop('kwargs', {}))
                        params.update({'save_intent': False})
                        _ = params.pop('intent_creator', 'Unknown')
                        canonical = eval(f"self.{method}(canonical=canonical, **params)", globals(), locals())
                except ValueError as ve:
                    raise ValueError(f"intent '{intent_level}', order '{order}', method '{method}' failed with: {ve}")
                except TypeError as te:
                    raise TypeError(f"intent '{intent_level}', order '{order}', method '{method}' failed with: {te}")
        if simulate:
            return pa.Table.from_pydict(col_sim)
        return canonical

    """
        PRIVATE METHODS SECTION
    """

    @staticmethod
    def _extract_value(value: [str, int, float]):
        if isinstance(value, str):
            if value.startswith('${') and value.endswith('}'):
                value = ConnectorContract.parse_environ(value)
                if value.isnumeric():
                    return int(value)
                elif value.replace('.', '', 1).isnumeric():
                    return float(value)
                else:
                    return str(value)
            else:
                return str(value)
        return value

    def _intent_builder(self, method: str, params: dict, exclude: list=None) -> dict:
        """builds the intent_params. Pass the method name and local() parameters
            Example:
                self._intent_builder(inspect.currentframe().f_code.co_name, **locals())

        :param method: the name of the method (intent). can use 'inspect.currentframe().f_code.co_name'
        :param params: the parameters passed to the method. use `locals()` in the caller method
        :param exclude: (optional) convenience parameter identifying param keys to exclude.
        :return: dict of the intent
        """
        return super()._intent_builder(method=method, params=params, exclude=exclude)

