import io
import pymupdf
import pymupdf4llm
import requests
import os
import pyarrow as pa
import pyarrow.parquet as pq
from ds_core.handlers.abstract_handlers import AbstractSourceHandler, AbstractPersistHandler
from ds_core.handlers.abstract_handlers import ConnectorContract, HandlerFactory

__author__ = 'Darryl Oatridge'


class KnowledgeSourceHandler(AbstractSourceHandler):
    """ This handler class uses pymupdf package. PyMuPDF is a Python binding for MuPDF, a
    lightweight PDF, XPS, and eBook reader.

        URI example
            uri = "<<filename>>" file_type='pdf', as_pages=False, as_markdown=False

        params:
            file_type: (optional) The file type being loaded. The default is 'pdf'
            as_pages: (optional) if the return should be in pages. The default is False
            as_markdown: (optional) if the return should be Markdown text. The default is False
    """

    def __init__(self, connector_contract: ConnectorContract):
        """ initialise the Handler passing the connector_contract dictionary """
        super().__init__(connector_contract)
        self._file_state = 0
        self._changed_flag = True

    def supported_types(self) -> list:
        """ The source types supported with this module"""
        return ['parquet', 'txt', 'pdf']

    def load_canonical(self, **kwargs) -> pa.Table:
        """ returns the canonical dataset based on the connector contract. """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The Connector Contract was not been set at initialisation or is corrupted")
        _cc = self.connector_contract
        load_params = kwargs
        load_params.update(_cc.kwargs)  # Update with any kwargs in the Connector Contract
        as_pages = load_params.get('as_pages', False)
        as_markdown = load_params.get('as_markdown', False)
        if load_params.get('file_type', False):
            file_type = load_params.pop('file_type', 'pdf')
            address = _cc.uri
        else:
            load_params.update(_cc.query)  # Update kwargs with those in the uri query
            _, _, _ext = _cc.address.rpartition('.')
            address = _cc.address
            file_type = load_params.pop('file_type', _ext if len(_ext) > 0 else 'pdf')
        self.reset_changed()
        # parquet
        if file_type.lower() in ['parquet']:
            if _cc.schema.startswith('http'):
                address = io.BytesIO(requests.get(address).content)
            return pq.read_table(address, **load_params)
        # pymupdf
        if _cc.schema.startswith('http'):
            request = requests.get(address)
            filestream = io.BytesIO(request.content)
            with pymupdf.open(stream=filestream, filetype=file_type) as doc:
                return self._get_text_from_doc(doc, as_markdown=as_markdown, as_pages=as_pages)
        else:
            with pymupdf.open(address) as doc:
                return self._get_text_from_doc(doc, as_markdown=as_markdown, as_pages=as_pages)

    def exists(self) -> bool:
        """ Returns True is the file exists """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The Connector Contract has not been set")
        _cc = self.connector_contract
        if _cc.schema.startswith('http'):
            r = requests.get(_cc.address)
            if r.status_code == 200:
                return True
        if os.path.exists(_cc.address):
            return True
        return False

    def has_changed(self) -> bool:
        """ returns the status of the change_flag indicating if the file has changed since last load or reset"""
        if not self.exists():
            return False
        # maintain the change flag
        _cc = self.connector_contract
        if _cc.schema.startswith('http') or _cc.schema.startswith('git'):
            if not isinstance(self.connector_contract, ConnectorContract):
                raise ValueError("The Pandas Connector Contract has not been set")
            module_name = 'requests'
            _address = _cc.address.replace("git://", "https://")
            if HandlerFactory.check_module(module_name=module_name):
                module = HandlerFactory.get_module(module_name=module_name)
                state = module.head(_address).headers.get('last-modified', 0)
            else:
                raise ModuleNotFoundError(f"The required module {module_name} has not been installed. Please pip "
                                          f"install the appropriate package in order to complete this action")
        else:
            state = os.stat(_cc.address).st_mtime_ns
        if state != self._file_state:
            self._changed_flag = True
            self._file_state = state
        return self._changed_flag

    def reset_changed(self, changed: bool = False):
        """ manual reset to say the file has been seen. This is automatically called if the file is loaded"""
        changed = changed if isinstance(changed, bool) else False
        self._changed_flag = changed

    @staticmethod
    def _get_text_from_doc(doc, as_markdown: bool, as_pages: bool) -> pa.Table:
        if as_pages:
            pages_and_texts = []
            if as_markdown:
                for idx, chunk in enumerate(pymupdf4llm.to_markdown(doc, page_chunks=True)):
                    text = chunk['text'].encode().decode()
                    pages_and_texts.append(
                        {"index": idx,
                         "page_number": chunk['metadata']['page'],
                         "page_char_count": len(text),
                         "page_word_count": len(text.split(" ")),
                         "page_sentence_count_raw": len(text.split(". ")),
                         "page_token_count": round(len(text) / 4),
                         "page_tables": chunk['tables'],
                         "text": text})
            else:
                for idx, page in enumerate(doc):
                    text = page.get_text().encode().decode()
                    pages_and_texts.append(
                        {"index": idx,
                         "page_number": page.number,
                         "page_char_count": len(text),
                         "page_word_count": len(text.split(" ")),
                         "page_sentence_count_raw": len(text.split(". ")),
                         "page_token_count": round(len(text) / 4),
                         "page_tables": [t.extract()
                                         for t in page.find_tables().tables] if page.find_tables().tables else [],
                         "text": text})
            return pa.Table.from_pylist(pages_and_texts)
        else:
            if as_markdown:
                text = pymupdf4llm.to_markdown(doc).encode().decode()
            else:
                text = chr(12).join([page.get_text() for page in doc]).encode().decode()
            t_array = pa.array([text], pa.string())
            i_array = pa.array([int(x) for x in range(len(t_array))])
            return pa.table([i_array, t_array], names=['index', 'text'])


class KnowledgePersistHandler(KnowledgeSourceHandler, AbstractPersistHandler):
    """ PyArrow read/write Persist Handler. """

    def persist_canonical(self, canonical: pa.Table, **kwargs) -> bool:
        """ persists the canonical dataset

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        _uri = self.connector_contract.uri
        return self.backup_canonical(uri=_uri, canonical=canonical, **kwargs)

    def backup_canonical(self, canonical: pa.Table, uri: str, **kwargs) -> bool:
        """ creates a backup of the canonical to an alternative URI

        Extra Parameters in the ConnectorContract kwargs:
            - file_type: (optional) the type of the source file. if not set, inferred from the file extension
            - write_params (optional) a dictionary of additional write parameters directly passed to 'write_' methods
        """
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        _cc = self.connector_contract
        _address = _cc.parse_address(uri=uri)
        persist_params = kwargs if isinstance(kwargs, dict) else _cc.kwargs
        persist_params.update(_cc.parse_query(uri=uri))
        _, _, _ext = _address.rpartition('.')
        if not self.connector_contract.schema.startswith('http'):
            _path, _ = os.path.split(_address)
            if len(_path) > 0 and not os.path.exists(_path):
                os.makedirs(_path)
        file_type = persist_params.pop('file_type', _ext if len(_ext) > 0 else 'parquet')
        write_params = persist_params.pop('write_params', {})
        # parquet
        if file_type.lower() in ['pq', 'parquet']:
            pq.write_table(canonical, _address, **write_params)
            return True
        # not found
        raise LookupError('The file format {} is not currently supported for write'.format(file_type))


    def remove_canonical(self) -> bool:
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        _cc = self.connector_contract
        if self.connector_contract.schema.startswith('http'):
            raise NotImplemented("Remove Canonical does not support {} schema based URIs".format(_cc.schema))
        if os.path.exists(_cc.address):
            os.remove(_cc.address)
            return True
        return False
