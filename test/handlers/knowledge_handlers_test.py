import unittest
import os
from pathlib import Path
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import pyarrow as pa

import pyarrow.compute as pc

from nn_rag.handlers.knowledge_handlers import KnowledgePersistHandler
from ds_core.handlers.abstract_handlers import ConnectorContract
from ds_core.properties.property_manager import PropertyManager

# Pandas setup
pd.set_option('max_colwidth', 320)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 99)
pd.set_option('expand_frame_repr', True)


class FeatureBuilderTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # clean out any old environments
        for key in os.environ.keys():
            if key.startswith('HADRON'):
                del os.environ[key]
        # Local Domain Contract
        os.environ['HADRON_PM_PATH'] = os.path.join('working', 'contracts')
        os.environ['HADRON_PM_TYPE'] = 'parquet'
        # Local Connectivity
        os.environ['HADRON_DEFAULT_PATH'] = Path('working/data').as_posix()
        # Specialist Component
        try:
            os.makedirs(os.environ['HADRON_PM_PATH'])
        except OSError:
            pass
        try:
            os.makedirs(os.environ['HADRON_DEFAULT_PATH'])
        except OSError:
            pass
        try:
            shutil.copytree('../_test_data', os.path.join(os.environ['PWD'], 'working/source'))
        except OSError:
            pass
        PropertyManager._remove_all()

    def tearDown(self):
        try:
            shutil.rmtree('working')
        except OSError:
            pass

    def test_parquet(self):
        tbl = get_table()
        uri = os.path.join(os.environ['HADRON_DEFAULT_PATH'], 'test.parquet')
        cc = ConnectorContract(uri, 'module_name', 'handler')
        handler = KnowledgePersistHandler(cc)
        handler.persist_canonical(tbl)
        result = handler.load_canonical()
        self.assertEqual(tbl.column_names, result.column_names)
        self.assertEqual(tbl.shape, result.shape)
        self.assertEqual(tbl.schema, result.schema)

    def test_pdf_https(self):
        # uri = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
        uri = 'https://www.europarl.europa.eu/doceo/document/TA-9-2024-0138_EN.pdf'
        cc = ConnectorContract(uri, 'module_name', 'handler')
        handler = KnowledgePersistHandler(cc)
        result = handler.load_canonical(file_type='pdf')
        print(result.shape)

    def test_txt(self):
        uri = 'https://filesamples.com/samples/document/txt/sample3.txt'
        cc = ConnectorContract(uri, 'module_name', 'handler')
        handler = KnowledgePersistHandler(cc)
        result = handler.load_canonical()
        c = result.column('text').combine_chunks()
        print(len(c))

    def test_write_parquet(self):
        uri = './working/data/tensor.parquet'
        cc = ConnectorContract(uri, 'module_name', 'handler')
        tbl = pa.table([ pa.array([1, 2, 3, 4, 5]),], names=["col1"])
        handler = KnowledgePersistHandler(cc)
        result = handler.persist_canonical(tbl)
        print(result)


    def test_raise(self):
        startTime = datetime.now()
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))
        print(f"Duration - {str(datetime.now() - startTime)}")



def get_table():
    num = pa.array([1.0, None, 5.0, -0.46421, 3.5, 7.233, -2], pa.float64())
    val = pa.array([1, 2, 3, 4, 5, 6, 7], pa.int64())
    date = pc.strptime(["2023-01-02 04:49:06", "2023-01-02 04:57:12", None, None, "2023-01-02 05:23:50", None, None],
                       format='%Y-%m-%d %H:%M:%S', unit='us')
    text = pa.array(["Blue", "Green", None, 'Red', 'Orange', 'Yellow', 'Pink'], pa.string())
    binary = pa.array([True, True, None, False, False, True, False], pa.bool_())
    cat = pa.array([None, 'M', 'F', 'M', 'F', 'M', 'M'], pa.string()).dictionary_encode()
    return pa.table([num, val, date, text, binary, cat], names=['num', 'int', 'date', 'text', 'bool', 'cat'])


if __name__ == '__main__':
    unittest.main()
