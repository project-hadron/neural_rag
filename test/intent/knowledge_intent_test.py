import unittest
import os
from pathlib import Path
import shutil
from datetime import datetime
import pandas as pd
import pyarrow as pa
import torch
from ds_core.properties.property_manager import PropertyManager
from nn_rag.components.commons import Commons
from nn_rag import Knowledge
from nn_rag.intent.knowledge_intent import KnowledgeIntent

# Pandas setup
pd.set_option('max_colwidth', 320)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 99)
pd.set_option('expand_frame_repr', True)


class KnowledgeIntentTest(unittest.TestCase):

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

    def test_chunk(self):
        kn = Knowledge.from_memory()
        tools: KnowledgeIntent = kn.tools
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
        arr1 = pa.array([1], type=pa.int64())
        arr2 = pa.array([text], pa.string())
        tbl = pa.table([arr1, arr2], names=['page_number', 'text'])
        text_chunks = tools.text_profiler(tbl)
        len(text_chunks)


    def test_splitting(self):
        kn = Knowledge.from_memory()
        tools: KnowledgeIntent = kn.tools
        # uri = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
        # kn.set_source_uri(uri)
        # tbl = kn.load_source_canonical(file_type='pdf')
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
        arr1 = pa.array([1], type=pa.int64())
        arr2 = pa.array([text], pa.string())
        tbl = pa.table([arr1, arr2], names=['page_number', 'text'])
        text_chunks = tools.text_profiler(tbl, header='text', num_sentence_chunk_size=3)
        # chunks
        chunks = tools.sentence_chunk(text_chunks, 'sentence_chunks')
        embedding = tools.chunk_embedding(chunks)
        print(embedding.shape)

        # tensor = torch.from_numpy(pa_tensor.to_numpy)

    def test_raise(self):
        startTime = datetime.now()
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))
        print(f"Duration - {str(datetime.now() - startTime)}")


def tprint(t: pa.table, headers: [str, list] = None, d_type: [str, list] = None, regex: [str, list] = None):
    _ = Commons.filter_columns(t.slice(0, 10), headers=headers, d_types=d_type, regex=regex)
    print(Commons.table_report(_).to_string())


if __name__ == '__main__':
    unittest.main()
