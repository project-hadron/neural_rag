import datetime
import time
import pandas as pd
import pyarrow as pa
from ds_core.handlers.abstract_handlers import ConnectorContract
from ds_core.components.abstract_component import AbstractComponent

from nn_rag.components.commons import Commons
from nn_rag.managers.controller_property_manager import ControllerPropertyManager
from nn_rag.intent.controller_intent import ControllerIntentModel
from nn_rag.components.discovery import DataDiscovery


# noinspection PyArgumentList
class Controller(AbstractComponent):
    """Controller Class is a special capability that controls a pipeline flow. It still
    inherits core AbstractComponent but not child AbstractCommonComponent that the other
    capabilities inherit from.
    """

    ASSET_BANK = 'https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/contracts/pyarrow/'
    TOY_SAMPLE = 'https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/datasets/toy_sample/'

    REPORT_USE_CASE = 'use_case'
    URI_PM_REPO = None

    def __init__(self, property_manager: ControllerPropertyManager, intent_model: ControllerIntentModel,
                 default_save=None, reset_templates: bool=None, template_path: str=None, template_module: str=None,
                 template_source_handler: str=None, template_persist_handler: str=None,
                 align_connectors: bool=None):
        """ Encapsulation class for the components set of classes

        :param property_manager: The contract property manager instance for this component
        :param intent_model: the model codebase containing the parameterizable intent
        :param default_save: The default behaviour of persisting the contracts:
                    if False: The connector contracts are kept in memory (useful for restricted file systems)
        :param reset_templates: (optional) reset connector templates from environ variables (see `report_environ()`)
        :param template_path: (optional) a template path to use if the environment variable does not exist
        :param template_module: (optional) a template module to use if the environment variable does not exist
        :param template_source_handler: (optional) a template source handler to use if no environment variable
        :param template_persist_handler: (optional) a template persist handler to use if no environment variable
        :param align_connectors: (optional) resets aligned connectors to the template
        """
        super().__init__(property_manager=property_manager, intent_model=intent_model, default_save=default_save,
                         reset_templates=reset_templates, template_path=template_path, template_module=template_module,
                         template_source_handler=template_source_handler,
                         template_persist_handler=template_persist_handler, align_connectors=align_connectors)

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, creator: str, uri_pm_repo: str=None, pm_file_type: str=None,
                 pm_module: str=None, pm_handler: str=None, pm_kwargs: dict=None, default_save=None,
                 reset_templates: bool=None, template_path: str=None, template_module: str=None,
                 template_source_handler: str=None, template_persist_handler: str=None, align_connectors: bool=None,
                 default_save_intent: bool=None, default_intent_level: bool=None, order_next_available: bool=None,
                 default_replace_intent: bool=None, has_contract: bool=None):
        """ Class Factory Method to instantiates the component's application. The Factory Method handles the
        instantiation of the Properties Manager, the Intent Model and the persistence of the uploaded properties.
        See class inline _docs for an example method

         :param task_name: The reference name that uniquely identifies a task or subset of the property manager
         :param uri_pm_path: A URI that identifies the resource path for the property manager.
         :param creator: A user name for this task activity.
         :param uri_pm_repo: (optional) A repository URI to initially load the property manager but not save to.
         :param pm_file_type: (optional) defines a specific file type for the property manager
         :param pm_module: (optional) the module or package name where the handler can be found
         :param pm_handler: (optional) the handler for retrieving the resource
         :param pm_kwargs: (optional) a dictionary of kwargs to pass to the property manager
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :param reset_templates: (optional) reset connector templates from environ variables. Default True
                                (see `report_environ()`)
         :param template_path: (optional) a template path to use if the environment variable does not exist
         :param template_module: (optional) a template module to use if the environment variable does not exist
         :param template_source_handler: (optional) a template source handler to use if no environment variable
         :param template_persist_handler: (optional) a template persist handler to use if no environment variable
         :param align_connectors: (optional) resets aligned connectors to the template. default Default True
         :param default_save_intent: (optional) The default action for saving intent in the property manager
         :param default_intent_level: (optional) the default level intent should be saved at
         :param order_next_available: (optional) if the default behaviour for the order should be next available order
         :param default_replace_intent: (optional) the default replace existing intent behaviour
         :param has_contract: (optional) indicates the instance should have a property manager domain contract
         :return: the initialised class instance
         """
        # save the controllers uri_pm_repo path
        if isinstance(uri_pm_repo, str):
            cls.URI_PM_REPO = uri_pm_repo
        pm_file_type = pm_file_type if isinstance(pm_file_type, str) else 'parquet'
        pm_module = pm_module if isinstance(pm_module, str) else cls.DEFAULT_MODULE
        pm_handler = pm_handler if isinstance(pm_handler, str) else cls.DEFAULT_PERSIST_HANDLER
        _pm = ControllerPropertyManager(task_name=task_name, creator=creator)
        _intent_model = ControllerIntentModel(property_manager=_pm, default_save_intent=default_save_intent,
                                              default_intent_level=default_intent_level,
                                              order_next_available=order_next_available,
                                              default_replace_intent=default_replace_intent)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, default_save=default_save,
                                 uri_pm_repo=uri_pm_repo, pm_file_type=pm_file_type, pm_module=pm_module,
                                 pm_handler=pm_handler, pm_kwargs=pm_kwargs, has_contract=has_contract)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save,
                   reset_templates=reset_templates, template_path=template_path, template_module=template_module,
                   template_source_handler=template_source_handler, template_persist_handler=template_persist_handler,
                   align_connectors=align_connectors)

    @classmethod
    def from_env(cls, task_name: str=None, default_save=None, reset_templates: bool=None, align_connectors: bool=None,
                 default_save_intent: bool=None, default_intent_level: bool=None, order_next_available: bool=None,
                 default_replace_intent: bool=None, uri_pm_repo: str=None, has_contract: bool=None,
                 **kwargs):
        """ Class Factory Method that builds the connector handlers taking the property contract path from
        the os.environ['HADRON_PM_PATH'] or, if not found, uses the system default,
                    for Linux and IOS '/tmp/components/contracts
                    for Windows 'os.environ['AppData']\\components\\contracts'
        The following environment variables can be set:
        'HADRON_PM_PATH': the property contract path, if not found, uses the system default
        'HADRON_PM_REPO': the property contract should be initially loaded from a read only repo site such as github
        'HADRON_PM_TYPE': a file type for the property manager. If not found sets as 'parquet'
        'HADRON_PM_MODULE': a default module package, if not set uses component default
        'HADRON_PM_HANDLER': a default handler. if not set uses component default

        This method calls to the Factory Method 'from_uri(...)' returning the initialised class instance

         :param task_name: (optional) The reference name that uniquely identifies the ledger. Defaults to 'primary'
         :param default_save: (optional) if the configuration should be persisted
         :param reset_templates: (optional) reset connector templates from environ variables. Default True
                                (see `report_environ()`)
         :param align_connectors: (optional) resets aligned connectors to the template. default Default True
         :param default_save_intent: (optional) The default action for saving intent in the property manager
         :param default_intent_level: (optional) the default level intent should be saved at
         :param order_next_available: (optional) if the default behaviour for the order should be next available order
         :param default_replace_intent: (optional) the default replace existing intent behaviour
         :param uri_pm_repo: The read only repo link that points to the raw data path to the contracts repo directory
         :param has_contract: (optional) indicates the instance should have a property manager domain contract
         :param kwargs: to pass to the property ConnectorContract as its kwargs
         :return: the initialised class instance
         """
        task_name = task_name if isinstance(task_name, str) else 'master'
        return super().from_env(task_name=task_name, default_save=default_save, reset_templates=reset_templates,
                                align_connectors=align_connectors, default_save_intent=default_save_intent,
                                default_intent_level=default_intent_level, order_next_available=order_next_available,
                                default_replace_intent=default_replace_intent, uri_pm_repo=uri_pm_repo,
                                has_contract=has_contract, **kwargs)

    @classmethod
    def scratch_pad(cls) -> ControllerIntentModel:
        """ A class method to use the Components intent methods as a scratch pad"""
        return super().scratch_pad()

    @property
    def tools(self) -> ControllerIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def register(self) -> ControllerIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def pm(self) -> ControllerPropertyManager:
        """The properties manager instance"""
        return self._component_pm

    def remove_all_tasks(self, save: bool=None):
        """removes all tasks"""
        for level in self.pm.get_intent():
            self.pm.remove_intent(level=level)
        self.pm_persist(save)

    def set_use_case(self, title: str=None, domain: str=None, overview: str=None, scope: str=None,
                     situation: str=None, opportunity: str=None, actions: str=None, project_name: str=None,
                     project_lead: str=None, project_contact: str=None, stakeholder_domain: str=None,
                     stakeholder_group: str=None, stakeholder_lead: str=None, stakeholder_contact: str=None,
                     save: bool=None):
        """ sets the use_case values. Only sets those passed

        :param title: (optional) the title of the use_case
        :param domain: (optional) the domain it sits within
        :param overview: (optional) a overview of the use case
        :param scope: (optional) the scope of responsibility
        :param situation: (optional) The inferred 'Why', 'What' or 'How' and predicted 'therefore can we'
        :param opportunity: (optional) The opportunity of the situation
        :param actions: (optional) the actions to fulfil the opportunity
        :param project_name: (optional) the name of the project this use case is for
        :param project_lead: (optional) the person who is project lead
        :param project_contact: (optional) the contact information for the project lead
        :param stakeholder_domain: (optional) the domain of the stakeholders
        :param stakeholder_group: (optional) the stakeholder group name
        :param stakeholder_lead: (optional) the stakeholder lead
        :param stakeholder_contact: (optional) contact information for the stakeholder lead
        :param save: (optional) if True, save to file. Default is True
        """
        self.pm.set_use_case(title=title, domain=domain, overview=overview, scope=scope, situation=situation,
                             opportunity=opportunity, actions=actions, project_name=project_name,
                             project_lead=project_lead, project_contact=project_contact,
                             stakeholder_domain=stakeholder_domain, stakeholder_group=stakeholder_group,
                             stakeholder_lead=stakeholder_lead, stakeholder_contact=stakeholder_contact)
        self.pm_persist(save=save)

    def load_source_canonical(self, reset_changed: bool=None, has_changed: bool=None, return_empty: bool=None,
                              **kwargs) -> pa.Table:
        """returns the contracted source data as a DataFrame

        :param reset_changed: (optional) resets the has_changed boolean to True
        :param has_changed: (optional) tests if the underline canonical has changed since last load else error returned
        :param return_empty: (optional) if has_changed is set, returns an empty canonical if set to True
        :param kwargs: arguments to be passed to the handler on load
        """
        return self.load_canonical(self.CONNECTOR_SOURCE, reset_changed=reset_changed, has_changed=has_changed,
                                   return_empty=return_empty, **kwargs)

    def load_canonical(self, connector_name: str, reset_changed: bool=None, has_changed: bool=None,
                       return_empty: bool=None, **kwargs) -> pa.Table:
        """returns the canonical of the referenced connector

        :param connector_name: the name or label to identify and reference the connector
        :param reset_changed: (optional) resets the has_changed boolean to True
        :param has_changed: (optional) tests if the underline canonical has changed since last load else error returned
        :param return_empty: (optional) if has_changed is set, returns an empty canonical if set to True
        :param kwargs: arguments to be passed to the handler on load
        """
        canonical = super().load_canonical(connector_name=connector_name, reset_changed=reset_changed,
                                           has_changed=has_changed, return_empty=return_empty, **kwargs)
        return canonical

    def load_persist_canonical(self, reset_changed: bool=None, has_changed: bool=None, return_empty: bool=None,
                               **kwargs) -> pa.Table:
        """loads the clean pandas.DataFrame from the clean folder for this contract

        :param reset_changed: (optional) resets the has_changed boolean to True
        :param has_changed: (optional) tests if the underline canonical has changed since last load else error returned
        :param return_empty: (optional) if has_changed is set, returns an empty canonical if set to True
        :param kwargs: arguments to be passed to the handler on load
        """
        return self.load_canonical(self.CONNECTOR_PERSIST, reset_changed=reset_changed, has_changed=has_changed,
                                   return_empty=return_empty, **kwargs)

    @staticmethod
    def quality_report(canonical: pa.Table, nulls_threshold: float = None,
                       dom_threshold: float = None,
                       cat_threshold: int = None, stylise: bool = None):
        """ Analyses a dataset, passed as a DataFrame and returns a quality summary

        :param canonical: The table to view.
        :param cat_threshold: (optional) The threshold for the max number of unique categories. Default is 60
        :param dom_threshold: (optional) The threshold limit of a dominant value. Default 0.98
        :param nulls_threshold: (optional) The threshold limit of a nulls value. Default 0.9
        :param stylise: (optional) if the output is stylised
        """
        stylise = stylise if isinstance(stylise, bool) else True
        return DataDiscovery.data_quality(canonical=canonical, nulls_threshold=nulls_threshold,
                                          dom_threshold=dom_threshold, cat_threshold=cat_threshold,
                                          stylise=stylise)

    @staticmethod
    def canonical_report(canonical: pa.Table, headers: [str, list] = None,
                         regex: [str, list] = None, d_types: list = None,
                         drop: bool = None, stylise: bool = None, display_width: int = None):
        """The Canonical Report is a data dictionary of the canonical providing a reference view of the dataset's
        attribute properties

        :param canonical: the table to view
        :param headers: (optional) specific headers to display
        :param regex: (optional) specify header regex to display. regex matching is done using the Google RE2 library.
        :param d_types: (optional) a list of pyarrow DataType e.g [pa.string(), pa.bool_()]
        :param drop: (optional) if the headers are to be dropped and the remaining to display
        :param stylise: (optional) if True present the report stylised.
        :param display_width: (optional) the width of the observational display
        """
        stylise = stylise if isinstance(stylise, bool) else True
        tbl = Commons.filter_columns(canonical, headers=headers, regex=regex, d_types=d_types,
                                     drop=drop)
        return DataDiscovery.data_dictionary(canonical=tbl, stylise=stylise,
                                             display_width=display_width)

    @staticmethod
    def schema_report(canonical: pa.Table, headers: [str, list] = None, regex: [str, list] = None,
                      d_types: list = None,
                      drop: bool = None, stylise: bool = True, table_cast: bool = None):
        """ presents the current canonical schema

        :param canonical: the table to view
        :param headers: (optional) specific headers to display
        :param regex: (optional) specify header regex to display. regex matching is done using the Google RE2 library.
        :param d_types: (optional) a list of pyarrow DataType e.g [pa.string(), pa.bool_()]
        :param drop: (optional) if the headers are to be dropped and the remaining to display
        :param stylise: (optional) if True present the report stylised.
        :param table_cast: (optional) if the column should try to be cast to its type
        """
        stylise = stylise if isinstance(stylise, bool) else True
        table_cast = table_cast if isinstance(table_cast, bool) else True
        tbl = Commons.filter_columns(canonical, headers=headers, regex=regex, d_types=d_types,
                                     drop=drop)
        return DataDiscovery.data_schema(canonical=tbl, table_cast=table_cast, stylise=stylise)

    @staticmethod
    def table_report(canonical: pa.Table, head: int=None):
        """ Creates a report from a pyarrow table in a tabular form

        :param canonical: the table to view
        :param head: The number of rows to show.
        """
        return Commons.table_report(canonical, head=head)

    def reset_use_case(self, save: bool=None):
        """resets the use_case back to its default values"""
        self.pm.reset_use_case()
        self.pm_persist(save)

    def report_use_case(self, stylise: bool=None):
        """ a report on the use_case set as part of the domain contract"""
        stylise = stylise if isinstance(stylise, bool) else True
        report = self.pm.report_use_case()
        report = pd.DataFrame(report, index=['values'])
        if stylise:
            report = report.transpose().reset_index()
            report.columns = ['use_case', 'values']
            return Commons.report(report, index_header='use_case')
        return pa.Table.from_pandas(report)

    def report_tasks(self, stylise: bool=True):
        """ generates a report for all the current component task"""
        report = pd.DataFrame.from_dict(data=self.pm.report_intent())
        intent_replace = {'feature_select': 'FeatureSelect', 'feature_engineer': 'FeatureEngineer',
                          'feature_transform': 'FeatureTransform', 'feature_build': 'FeatureBuild',
                          'feature_predict': 'FeaturePredict'}
        report['component'] = report.intent.replace(to_replace=intent_replace)
        report = report.loc[:, ['level', 'order', 'component', 'parameters', 'creator']]
        if stylise:
            return Commons.report(report, index_header='level')
        return pa.Table.from_pandas(report)

    def report_run_book(self, stylise: bool=True):
        """ generates a report on all the intent actions"""
        report = pd.DataFrame(self.pm.report_run_book())
        report = report.explode(column='run_book', ignore_index=True)
        report = report.join(pd.json_normalize(report['run_book'])).drop(columns=['run_book']).fillna('')
        if stylise:
            return Commons.report(report, index_header='name')
        return pa.Table.from_pandas(report)

    def add_run_book(self, run_levels: [str, list], book_name: str=None, save: bool=None):
        """ sets a named run book, the run levels are a list of levels and the order they are run in

        :param run_levels: the name or list of levels to be run
        :param book_name: (optional) the name of the run_book. defaults to 'primary_run_book'
        :param save: (optional) override of the default save action set at initialisation.
       """
        book_name = book_name if isinstance(book_name, str) else self.pm.PRIMARY_RUN_BOOK
        for run_level in Commons.list_formatter(run_levels):
            self.add_run_book_level(run_level=run_level, book_name=book_name)
        return

    def add_run_book_level(self, run_level: str, book_name: str=None, source: str=None, persist: [str, list]=None,
                           save: bool=None):
        """ adds a single runlevel to the end of a run_book. If the name already exists it will be replaced. When
        designing the runbook this gives easier implementation of the data handling specific for this runbook

        :param run_level: the run_level to add.
        :param book_name: (optional) the name of the run_book. defaults to 'primary_run_book'
        :param source: (optional) the intent level source
        :param persist: (optional) the intent level persist
        :param save: (optional) override of the default save action set at initialisation.
       """
        book_name = book_name if isinstance(book_name, str) else self.pm.PRIMARY_RUN_BOOK
        run_level = Commons.param2dict(run_level=run_level, source=source, persist=persist)
        if self.pm.has_run_book(book_name=book_name):
            run_levels = self.pm.get_run_book(book_name=book_name)
            for i in range(len(run_levels)):
                if isinstance(run_levels[i], str) and run_levels[i] == run_level:
                    run_levels[i].remove(run_level)
                elif isinstance(run_levels[i], dict) and run_levels[i].get('run_level') == run_level.get('run_level'):
                    del run_levels[i]
            run_levels.append(run_level)
        else:
            run_levels = [run_level]
        self.pm.set_run_book(book_name=book_name, run_levels=run_levels)
        self.pm_persist(save)


    def report_intent(self, levels: [str, int, list]=None, stylise: bool = True):
        """ generates a report on all the intent

        :param levels: (optional) a filter on the levels. passing a single value will report a single parameterised view
        :param stylise: (optional) returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        if isinstance(levels, (int, str)):
            report = pd.DataFrame.from_dict(data=self.pm.report_intent_params(level=levels))
            if stylise:
                return Commons.report(report, index_header='order')
        else:
            report = pd.DataFrame.from_dict(data=self.pm.report_intent(levels=levels))
            if stylise:
                return Commons.report(report, index_header='level')
        return pa.Table.from_pandas(report)

    def report_notes(self, catalog: [str, list]=None, labels: [str, list]=None, regex: [str, list]=None,
                     re_ignore_case: bool=False, stylise: bool=True, drop_dates: bool=False):
        """ generates a report on the notes

        :param catalog: (optional) the catalog to filter on
        :param labels: (optional) s label or list of labels to filter on
        :param regex: (optional) a regular expression on the notes
        :param re_ignore_case: (optional) if the regular expression should be case sensitive
        :param stylise: (optional) returns a stylised dataframe with formatting
        :param drop_dates: (optional) excludes the 'date' column from the report
        :return: pd.Dataframe
        """
        stylise = True if not isinstance(stylise, bool) else stylise
        drop_dates = False if not isinstance(drop_dates, bool) else drop_dates
        report = self.pm.report_notes(catalog=catalog, labels=labels, regex=regex, re_ignore_case=re_ignore_case,
                                      drop_dates=drop_dates)
        report = pd.DataFrame.from_dict(data=report)
        if stylise:
            return Commons.report(report, index_header='section', large_font='label')
        return pa.Table.from_pandas(report)

    def run_controller(self, run_book: [str, list, dict]=None, repeat: int=None, sleep: int=None, run_time: int=None,
                       source_check_uri: str=None, run_cycle_report: str=None):
        """ Runs the components pipeline based on the runbook instructions. The run_book is a pre-registered
        Controller run_book names to execute. If no run book is given, default values are substituted, finally taking
        the intent list if all else fails.

        The run_cycle_report automatically generates the connector contract with the name 'run_cycle_report'. To reload
        the report for observation use the controller method 'load_canonical(...) passing the name 'run_cycle_report'.

        :param run_book: (optional) a run_book reference, a list of task names (intent levels)
        :param repeat: (optional) the number of times this intent should be repeated. None or -1 -> never, 0 -> forever
        :param sleep: (optional) number of seconds to sleep before repeating
        :param run_time: (optional) number of seconds to run the controller using repeat and sleep cycles time is up
        :param source_check_uri: (optional) The source uri to check for change since last controller instance cycle
        :param run_cycle_report: (optional) a full name for the run cycle report
        """
        if isinstance(run_cycle_report, str):
            self.add_connector_persist(connector_name='run_cycle_report', uri_file=run_cycle_report)
        df_report = pd.DataFrame(columns=['time', 'text'])
        if not self.pm.has_intent():
            return
        # tidy run_book
        run_book = Commons.list_formatter(run_book)
        if not run_book:
            if self.pm.has_run_book(self.pm.PRIMARY_RUN_BOOK):
                run_book = self.pm.get_run_book(self.pm.PRIMARY_RUN_BOOK)
            elif self.pm.has_intent(self.pm.DEFAULT_INTENT_LEVEL):
                self.add_run_book_level(run_level=self.pm.DEFAULT_INTENT_LEVEL, book_name=self.pm.PRIMARY_RUN_BOOK)
                run_book = self.pm.get_run_book(self.pm.PRIMARY_RUN_BOOK)
            else:
                for level in self.pm.get_intent().keys():
                    self.add_run_book_level(run_level=level, book_name=self.pm.PRIMARY_RUN_BOOK)
                run_book = self.pm.get_run_book(self.pm.PRIMARY_RUN_BOOK)
        run_dict = []
        for book in run_book:
            if isinstance(book, str):
                if self.pm.has_run_book(book):
                    for item in self.pm.get_run_book(book):
                        run_dict.append(item)
                else:
                    run_dict.append({'run_level': f'{book}'})
            else:
                run_dict.append(book)
        handler = None
        if isinstance(source_check_uri, str):
            self.add_connector_uri(connector_name='source_check_uri', uri=source_check_uri)
            handler = self.pm.get_connector_handler(connector_name='source_check_uri')
        run_time = run_time if isinstance(run_time, int) else 0
        if run_time > 0 and not isinstance(sleep, int):
            sleep = 1
        end_time = datetime.datetime.now() + datetime.timedelta(seconds=run_time)
        repeat = repeat if isinstance(repeat, int) and repeat > 0 else 1
        run_count = 0
        while True: # run_time always runs once
            if isinstance(run_cycle_report, str):
                df_report.loc[len(df_report.index)] = [datetime.datetime.now(), f'starting pipeline -{run_count}-']
            for count in range(repeat):
                if isinstance(run_cycle_report, str):
                    df_report.loc[len(df_report.index)] = [datetime.datetime.now(), f'starting cycle -{count}-']
                if handler and handler.exists():
                    if handler.has_changed():
                        handler.reset_changed(False)
                    else:
                        if isinstance(run_cycle_report, str):
                            df_report.loc[len(df_report.index)] = [datetime.datetime.now(), 'Source has not changed']
                        if isinstance(sleep, int) and count < repeat - 1:
                            time.sleep(sleep)
                        continue
                for book in run_dict:
                    run_level = book.get('run_level')
                    source = book.get('source', self.CONNECTOR_SOURCE)
                    persist = book.get('persist', self.CONNECTOR_PERSIST)
                    if isinstance(run_cycle_report, str):
                        df_report.loc[len(df_report.index)] = [datetime.datetime.now(), f"running: '{run_level}'"]
                    # run level
                    shape = self.intent_model.run_intent_pipeline(run_level=run_level, source=source, persist=persist,
                                                                  controller_repo=self.URI_PM_REPO)
                    if isinstance(run_cycle_report, str):
                        df_report.loc[len(df_report.index)] = [datetime.datetime.now(), f'outcome shape: {shape}']
                if isinstance(run_cycle_report, str):
                    df_report.loc[len(df_report.index)] = [datetime.datetime.now(), 'cycle complete']
                if isinstance(sleep, int) and count < repeat-1:
                    if isinstance(run_cycle_report, str):
                        df_report.loc[len(df_report.index)] = [datetime.datetime.now(), f'sleep for {sleep} seconds']
                    time.sleep(sleep)
            if isinstance(run_cycle_report, str):
                run_count += 1
            if end_time < datetime.datetime.now():
                break
            else:
                if isinstance(run_cycle_report, str):
                    df_report.loc[len(df_report.index)] = [datetime.datetime.now(), f'sleep for {sleep} seconds']
                time.sleep(sleep)
        if isinstance(run_cycle_report, str):
            df_report.loc[len(df_report.index)] = [datetime.datetime.now(), 'pipeline complete']
            _ = pa.Table.from_pandas(df_report, preserve_index=False)
            self.save_canonical(connector_name='run_cycle_report', canonical=_)
        return

    @staticmethod
    def runbook_script(task: str, source: str=None, persist: str=None, drop: bool=None) -> dict:
        """ a utility method to help build feature conditions by aligning method parameters with dictionary format.
        This also sets default, in-memory and connector contracts specific for a task.
        if not specified, source or persist will use the default values. '${<connector>}' will use the environment
        variable is a connector contract of that name that must exist in the task connectors.

        :param task: the task name (intent level) name this runbook is applied too or a number if synthetic generation.
        :param source: (optional) a task name indicating where the source of this task will come from.
        :param persist: (optional) a task name indicating where the persist of this task will go to.
        :param drop: (optional) if true indicates in-memory canonical should be dropped once read.
        :return: dictionary of the parameters
        """
        source = source if isinstance(source, str) else Controller.CONNECTOR_SOURCE
        source = ConnectorContract.parse_environ(source) if source.startswith('${') and source.endswith('}') else source
        persist = persist if isinstance(persist, str) else Controller.CONNECTOR_PERSIST
        persist = ConnectorContract.parse_environ(persist) if persist.startswith('${') and persist.endswith('}') else persist
        drop = drop if isinstance(drop, bool) else True
        return Commons.param2dict(**locals())

    @staticmethod
    def get_pkg_root():
        return 'nn_rag'
