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

from ds_core.properties.abstract_properties import AbstractPropertyManager


class KnowledgePropertyManager(AbstractPropertyManager):

    def __init__(self, task_name: str, creator: str):
        """initialises the property manager.

        :param task_name: the name of the task name within the property manager
        :param creator: a username of this instance
        """
        root_keys = []
        knowledge_keys = ['describe']
        super().__init__(task_name=task_name, root_keys=root_keys, knowledge_keys=knowledge_keys, creator=creator)

    @staticmethod
    def get_pkg_root():
        return 'nn_rag'

