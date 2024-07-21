import unittest
import os
from pathlib import Path
import shutil
import ast
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ds_core.properties.property_manager import PropertyManager
from ds_capability import *
from ds_capability.components.commons import Commons

from nn_rag import Knowledge, Retrieval
from nn_rag.intent.retrieval_intent import RetrievalIntent

# Pandas setup
pd.set_option('max_colwidth', 320)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 99)
pd.set_option('expand_frame_repr', True)


class RetrievalTest(unittest.TestCase):

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

    def test_simple_query(self):
        set_kb()
        rag = Retrieval.from_memory()
        tools: RetrievalIntent = rag.tools
        rag.set_source_uri(uri="chroma:///")
        result = tools.query_similarity('you took ages')
        print(rag.table_report(result, head=5).to_string())

    def test_reranker_query(self):
        set_kb()
        rag = Retrieval.from_memory()
        tools: RetrievalIntent = rag.tools
        rag.set_source_uri(uri="chroma:///")
        result = tools.query_reranker('you took ages')
        print(rag.table_report(result).to_string())

    def test_raise(self):
        startTime = datetime.now()
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))
        print(f"Duration - {str(datetime.now() - startTime)}")


def set_kb():
    kn = Knowledge.from_memory()
    kn.set_persist_uri('chroma:///')
    text = ('You took too long. You are not easy to deal with. Payment Failure/Incorrect Payment. You provided '
            'me with incorrect information. Unhappy with delay. Unsuitable advice. You never answered my question. '
            'You did not understand my needs. I have been mis-sold. My details are not accurate. You have asked '
            'for too much information. You were not helpful. Payment not generated/received by customer. You did '
            'not keep me updated. Incorrect information given. The performance of my product was poor. No reply '
            'to customer contact. Requested documentation not issued. You did not explain the terms & conditions. '
            'Policy amendments not carried out. You did not explain the next steps/process to me. I cannot '
            'understand your letter/comms. Standard letter inappropriate. Customer payment processed incorrectly. '
            'All points not addressed. Could not understand the agent. Issue with terms and conditions. Misleading '
            'information. I can not use the customer portal. your customer portal is unhelpful')
    arr = pa.array([text], pa.string())
    tbl = pa.table([arr], names=['text'])
    tbl = kn.tools.text_to_sentences(tbl)
    kn.save_persist_canonical(tbl)

def tprint(t: pa.table, headers: [str, list] = None, d_type: [str, list] = None, regex: [str, list] = None):
    _ = Commons.filter_columns(t.slice(0, 10), headers=headers, d_types=d_type, regex=regex)
    print(Commons.table_report(_).to_string())


if __name__ == '__main__':
    unittest.main()
