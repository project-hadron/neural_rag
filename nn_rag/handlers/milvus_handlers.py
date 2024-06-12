import os
from ds_core.handlers.abstract_handlers import AbstractSourceHandler, ConnectorContract
from ds_core.handlers.abstract_handlers import HandlerFactory, AbstractPersistHandler
from sentence_transformers import SentenceTransformer
import pyarrow as pa

__author__ = 'Darryl Oatridge'


class MilvusSourceHandler(AbstractSourceHandler):
    """ This handler class uses both SQLAlchemy and pymysql. Together, SQLAlchemy and pymysql provide a powerful
    toolset for working with databases and faster connectivity. SQLAlchemy allows developers to interact with MySQL
    using Python code, while MySQL provides the database functionality needed to store and retrieve data efficiently.

        URI example
            uri = "milvus://host:port/collection=&partition"

        params:
            collection: The name of the collection
            partition: The name of a partition

        Environment:
            MILVUS_EMBEDDING_NAME
            MILVUS_EMBEDDING_DEVICE
            MILVUS_EMBEDDING_BATCH_SIZE
            MILVUS_EMBEDDING_DIM
            MILVUS_RESPONSE_LIMIT
            MILVUS_INDEX_METRIC
            MILVUS_DOC_REF
    """

    def __init__(self, connector_contract: ConnectorContract):
        """ initialise the Handler passing the Connector Contract """
        # required module import
        self.pymilvus = HandlerFactory.get_module('pymilvus')
        super().__init__(connector_contract)
        # reset to use dialect
        _kwargs = {**self.connector_contract.kwargs, **self.connector_contract.query}
        _embedding_name = os.environ.get('MILVUS_EMBEDDING_NAME', _kwargs.pop('embedding', 'all-mpnet-base-v2'))
        _device = os.environ.get('MILVUS_EMBEDDING_DEVICE', _kwargs.pop('device', 'cpu'))
        self._batch_size = int(os.environ.get('MILVUS_EMBEDDING_BATCH_SIZE', _kwargs.pop('batch_size', '32')))
        self._dimensions = int(os.environ.get('MILVUS_EMBEDDING_DIM', _kwargs.pop('dim', '768')))
        self._response_limit = int(os.environ.get('MILVUS_RESPONSE_LIMIT', _kwargs.pop('response_limit', '3')))
        self._metric_type = os.environ.get('MILVUS_INDEX_METRIC', _kwargs.pop('index_metric', 'L2'))
        self._doc_ref = os.environ.get('MILVUS_DOC_REF', _kwargs.pop('document', 'general'))
        self._collection_name = _kwargs.pop('collection', "default")
        description = "Standard Schema"
        # embedding model
        self._embedding_model = SentenceTransformer(model_name_or_path=_embedding_name, device=_device)
        # Start the server
        self.pymilvus.connections.connect(host=connector_contract.hostname, port=connector_contract.port)
        # Create the collection
        fields = [
            self.pymilvus.FieldSchema(name="id", dtype=self.pymilvus.DataType.VARCHAR, auto_id=False, is_primary=True, max_length=100),
            self.pymilvus.FieldSchema(name="source", dtype=self.pymilvus.DataType.VARCHAR, max_length=500),
            self.pymilvus.FieldSchema(name="embeddings", dtype=self.pymilvus.DataType.FLOAT_VECTOR, dim=self._dimensions)
        ]
        # schema
        schema = self.pymilvus.CollectionSchema(fields=fields)
        # collection
        self._collection = self.pymilvus.Collection(self._collection_name, schema)
        # index
        index = {'metric_type': self._metric_type, "params": {}}
        self._collection.create_index("embeddings", index)
        self._changed_flag = True

    def supported_types(self) -> list:
        """ The source types supported with this module"""
        return ['milvus']

    def exists(self) -> bool:
        """If the table exists"""
        return True

    def has_changed(self) -> bool:
        """ if the table has changed. Only works with certain implementations"""
        return self._changed_flag

    def reset_changed(self, changed: bool = False):
        """ manual reset to say the table has been seen. This is automatically called if the file is loaded"""
        changed = changed if isinstance(changed, bool) else False
        self._changed_flag = changed

    def load_canonical(self, query: str, **kwargs) -> dict:
        """ returns the canonical dataset based on the Connector Contract """
        if not isinstance(self.connector_contract, ConnectorContract):
            raise ValueError("The Connector Contract is not valid")
        self._collection.load()
        # embedding
        query_vector = self._embedding_model.encode(query)
        # search
        params = {"metric_type": self._metric_type, "params": {"nprobe": 10}}
        results = self._collection.search([query_vector], "embeddings", params, limit=self._response_limit, output_fields=["source"])
        self._collection.release()
        ids = pa.array(results[0].ids, pa.string())
        distances = pa.array(results[0].distances, pa.float32())
        entities = pa.array([x.entity.to_dict()['entity']['source'] for x in results[0]], pa.string())
        return pa.table([ids, distances, entities], names=['id', 'distance', 'source'])

class MilvusPersistHandler(MilvusSourceHandler, AbstractPersistHandler):
    # a Milvus persist handler

    def persist_canonical(self, canonical: pa.Table, **kwargs) -> bool:
        """ persists the canonical dataset"""
        return self.backup_canonical(canonical=canonical, **kwargs)

    def backup_canonical(self, canonical: pa.Table, **kwargs) -> bool:
        """ creates a backup of the canonical to an alternative table  """
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        _params = kwargs
        chunks = canonical.to_pylist()
        text_chunks = [item["chunk_text"] for item in chunks]
        embeddings = self._embedding_model.encode(text_chunks, batch_size=self._batch_size)
        data = [
            [f"{str(self._doc_ref)}_{str(i)}" for i in range(len(text_chunks))],
            text_chunks,
            embeddings
        ]
        self._collection.load()
        self._collection.upsert(data=data)
        self._collection.release()
        return

    def remove_canonical(self) -> bool:
        """removes the table and content"""
        if not isinstance(self.connector_contract, ConnectorContract):
            return False
        _cc = self.connector_contract
        return True
