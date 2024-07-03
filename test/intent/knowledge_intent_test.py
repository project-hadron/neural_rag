import unittest
import os
from pathlib import Path
import shutil
from datetime import datetime
import pandas as pd
import pyarrow as pa
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

    def test_str_remove_text_index(self):
        kn = Knowledge.from_memory()
        tools: KnowledgeIntent = kn.tools
        text = ('You took too long. You are not easy to deal with. Payment Failure/Incorrect Payment.\n\nYou provided '
                'me with incorrect information. Unhappy with delay.\n\nUnsuitable advice. You never answered my question.'
                'You did not understand my needs.\n\nI have been mis-sold. My details are not accurate.')
        arr = pa.array([text], pa.string())
        tbl = pa.table([arr], names=['text'])
        result = tools.text_to_paragraphs(tbl)
        print(kn.table_report(result, head=5).to_string())
        result = tools.filter_on_mask(result, indices=[0, (2, 7)])
        print(kn.table_report(result, head=5).to_string())

    def test_str_remove_text_pattern(self):
        kn = Knowledge.from_memory()
        tools: KnowledgeIntent = kn.tools
        text = ('You took too long. You are not easy to deal with. Payment Failure/Incorrect Payment.\n\nYou provided '
                'me with incorrect information. Unhappy with delay.\n\nUnsuitable advice. You never answered my question.'
                'You did not understand my needs.\n\nI have been mis-sold. My details are not accurate.')
        arr = pa.array([text], pa.string())
        tbl = pa.table([arr], names=['text'])
        result = tools.text_to_paragraphs(tbl)
        print(kn.table_report(result, head=5).to_string())
        result = tools.filter_on_mask(result, pattern='^You.*(You|Unhappy)')
        print(kn.table_report(result, head=5).to_string())


    def test_text_to_paragraph(self):
        kn = Knowledge.from_memory()
        tools: KnowledgeIntent = kn.tools
        uri = "https://assets.circle.so/kvx4ix1f5ctctk55daheobna46hf"
        tbl = kn.set_source_uri(uri).load_source_canonical()
        # text = ('You took too long. You are not easy to deal with. Payment Failure/Incorrect Payment. You provided '
        #         'me with incorrect information. Unhappy with delay. Unsuitable advice. You never answered my question.\n\n'
        #         'You did not understand my needs. I have been mis-sold. My details are not accurate. You have asked '
        #         'for too much information. You were not helpful. Payment not generated/received by customer. You did '
        #         'not keep me updated. Incorrect information given. The performance of my product was poor.\n\n No reply '
        #         'to customer contact. Requested documentation not issued. You did not explain the terms & conditions.\n\n'
        #         'Policy amendments not carried out. You did not explain the next steps/process to me. I cannot '
        #         'understand your letter/comms. Standard letter inappropriate. Customer payment processed incorrectly.\n\n'
        #         'All points not addressed. Could not understand the agent. Issue with terms and conditions. Misleading '
        #         'information. I can not use the customer portal. your customer portal is unhelpful.')
        # arr = pa.array([text], pa.string())
        # tbl = pa.table([arr], names=['text'])
        result = tools.text_to_paragraphs(tbl)
        print(kn.table_report(result, head=5).to_string())

    def test_text_to_sentence(self):
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
        arr = pa.array([text], pa.string())
        tbl = pa.table([arr], names=['text'])
        result = tools.text_to_sentences(tbl)
        print(kn.table_report(result, head=5).to_string())

    def test_text_to_sentence_max(self):
        kn = Knowledge.from_env('tester', has_contract=False)
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
        arr = pa.array([text], pa.string())
        tbl = pa.table([arr], names=['text'])
        # uri = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
        # tbl = kn.set_source_uri(uri, file_type='pdf').load_source_canonical()
        result =  tools.text_to_sentences(tbl, max_char_size=100)
        tprint(result, headers=['sentence_score', 'char_count', 'word_count'])


    def test_text_chunk(self):
        kn = Knowledge.from_memory()
        tools: KnowledgeIntent = kn.tools
        uri = "https://assets.circle.so/kvx4ix1f5ctctk55daheobna46hf"
        kn.set_source_uri(uri)
        tbl = kn.load_source_canonical(file_type='pdf')
        # text = ('You took too long. You took too long. You are not easy to deal with. Payment Failure/Incorrect Payment. You provided '
        #         'me with incorrect information. Unhappy with delay. Unsuitable advice. You never answered my question. '
        #         'You did not understand my needs. I have been mis-sold. My details are not accurate. You have asked '
        #         'for too much information. You were not helpful. Payment not generated/received by customer. You did '
        #         'not keep me updated. Incorrect information given. The performance of my product was poor. No reply '
        #         'to customer contact. Requested documentation not issued. You did not explain the terms & conditions. '
        #         'Policy amendments not carried out. You did not explain the next steps/process to me. I cannot '
        #         'understand your letter/comms. Standard letter inappropriate. Customer payment processed incorrectly. '
        #         'All points not addressed. Could not understand the agent. Issue with terms and conditions. Misleading '
        #         'information. I can not use the customer portal. your customer portal is unhelpful')
        # arr = pa.array([text], pa.string())
        # tbl = pa.table([arr], names=['text'])
        sentences = tools.text_to_sentences(tbl)
        result = tools.text_to_chunks(sentences, char_chunk_size=500, overlap=10)
        print(kn.table_report(result).to_string())

    def test_text_chunk_semantic(self):
        kn = Knowledge.from_memory()
        tools: KnowledgeIntent = kn.tools
        # uri = "https://www.europarl.europa.eu/doceo/document/TA-9-2024-0138_EN.pdf"
        # kn.set_source_uri(uri)
        # tbl = kn.load_source_canonical(file_type='pdf')
        text = ('You took too long. You took too long. You are not easy to deal with. Payment Failure/Incorrect Payment. You provided '
                'me with incorrect information. Unhappy with delay. Unhappy with delay. Unsuitable advice. You never answered my question. '
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
        sentences = tools.text_to_sentences(tbl)
        result = tools.text_to_chunks(sentences, char_chunk_size=100)
        print(kn.table_report(result).to_string())

    def test_embedding(self):
        kn = Knowledge.from_memory()
        kn.set_source_uri("milvus://localhost:19530/rai")
        kn.set_persist_uri("milvus://localhost:19530/rai")
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
        arr = pa.array([text], pa.string())
        tbl = pa.table([arr], names=['text'])
        sentences = tools.text_to_sentences(tbl)
        chunks = tools.text_to_chunks(sentences)
        # save
        kn.save_persist_canonical(chunks)
        result = kn.load_persist_canonical(query='long wait')
        print(kn.table_report(result).to_string())
        # kn.remove_canonical(kn.CONNECTOR_PERSIST)
        # result = kn.load_persist_canonical(query='long wait')
        # print(kn.table_report(result).to_string())

    def test_text_from_load(self):
        kn = Knowledge.from_memory()
        tools: KnowledgeIntent = kn.tools
        uri = '../../jupyter/knowledge/hadron/source/Global-Index-1st-Edition-Report.pdf'
        kn.set_source_uri(uri)
        tbl = kn.load_source_canonical(file_type='pdf', as_pages=True, as_markdown=False)
        print(kn.table_report(tbl, headers='text', drop=True).to_string())

    def test_text_join(self):
        kn = Knowledge.from_memory()
        tools: KnowledgeIntent = kn.tools
        uri = "https://assets.circle.so/kvx4ix1f5ctctk55daheobna46hf"
        kn.set_source_uri(uri)
        tbl = kn.load_source_canonical(file_type='pdf', as_pages=True)
        result = tools.text_join(tbl)
        print(kn.table_report(result, headers='text', drop=True).to_string())

    def test_raise(self):
        startTime = datetime.now()
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))
        print(f"Duration - {str(datetime.now() - startTime)}")


def get_table():
    n_legs = pa.array([2, 4, 5, 100])
    animals = pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"])
    names = ["n_legs", "animals"]
    return pa.Table.from_arrays([n_legs, animals], names=names)


def tprint(t: pa.table, top: int=None, headers: [str, list]=None, d_type: [str, list]=None, regex: [str, list]=None):
    top = top if isinstance(top, int) else 10
    _ = Commons.filter_columns(t.slice(0, top), headers=headers, d_types=d_type, regex=regex)
    print(Commons.table_report(_).to_string())


if __name__ == '__main__':
    unittest.main()
