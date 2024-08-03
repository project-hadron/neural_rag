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
import inspect
import torch
import pyarrow as pa
import pyarrow.compute as pc
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

from nn_rag.components.commons import Commons
from nn_rag.intent.abstract_generator_intent import AbstractGeneratorIntentModel


class GeneratorIntent(AbstractGeneratorIntentModel):

    CONNECTOR_SOURCE = 'primary_source'

    def query_similarity(self, query: str, limit: int=None, save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                         replace_intent: bool=None, remove_duplicates: bool=None):
        """

         :param query: text query to run against the list of tensor embeddings
         :param limit: (optional) the number of items to return from the results of the query
         :param save_intent: (optional) if the intent contract should be saved to the property manager
         :param intent_level: (optional) the intent name that groups intent to create a column
         :param intent_order: (optional) the order in which each intent should run.
                     - If None: default's to -1
                     - if -1: added to a level above any current instance of the intent section, level 0 if not found
                     - if int: added to the level specified, overwriting any that already exist

         :param replace_intent: (optional) if the intent method exists at the level, or default level
                     - True - replaces the current intent method with the new
                     - False - leaves it untouched, disregarding the new intent

         :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
         """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        query = self._extract_value(query)
        limit = limit if isinstance(limit, int) else 10
        if self._pm.has_connector(self.CONNECTOR_SOURCE):
            handler = self._pm.get_connector_handler(self.CONNECTOR_SOURCE)
            return handler.load_canonical(query=query, limit=limit)
        raise ValueError(f"The source connector has been set")

    def query_reranker(self, query: str, bi_limit: int=None, cross_limit: int=None,
                       save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                       replace_intent: bool=None, remove_duplicates: bool=None):
        """

         :param query: bool text query to run against the list of tensor embeddings
         :param bi_limit: (optional) bi encoder limits for results of the query
         :param cross_limit: (optional) cross encoder limits for results of the query
         :param save_intent: (optional) if the intent contract should be saved to the property manager
         :param intent_level: (optional) the intent name that groups intent to create a column
         :param intent_order: (optional) the order in which each intent should run.
                     - If None: default's to -1
                     - if -1: added to a level above any current instance of the intent section, level 0 if not found
                     - if int: added to the level specified, overwriting any that already exist

         :param replace_intent: (optional) if the intent method exists at the level, or default level
                     - True - replaces the current intent method with the new
                     - False - leaves it untouched, disregarding the new intent

         :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
         """
        # intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # remove intent params
        query = self._extract_value(query)
        bi_limit = bi_limit if isinstance(bi_limit, int) else 20
        cross_limit = cross_limit if isinstance(cross_limit, int) else 5
        device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        bi_answer = self.query_similarity(query=query, limit=bi_limit)
        # cross encoder
        model = CrossEncoder("cross-encoder/stsb-roberta-base", device=device)
        sentence_pairs = [[query, s] for s in bi_answer['text'].to_pylist()]
        ce_scores = model.predict(sentence_pairs)
        tbl = Commons.table_append(bi_answer, pa.table([pa.array(ce_scores, pa.float32()), bi_answer['text']], names=["score", "text"]))
        sorted_indices = pc.sort_indices(tbl, sort_keys=[("score", "descending")])
        return tbl.take(sorted_indices).slice(0, 5)

    def model_instantiate(self, model_id: str, quantize: str = None,
                          save_intent: bool = None, intent_level: [int, str] = None, intent_order: int = None,
                          replace_intent: bool = None, remove_duplicates: bool = None) -> pa.Table:
        """"""
        # intent recipie options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # set device
        _device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        # GPU optimization
        if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"
        # quantization
        _bnb_config = None
        if isinstance(quantize, str) and quantize in ['4bit', '8bit']:
            _bnb_config = BitsAndBytesConfig(
                load_in_4bit=True if quantize == '4bit' else False,
                load_in_8bit=True if quantize == '8bit' else False,
            )
        # instantiate the tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=_bnb_config,
            device_map=_device,
            attn_implementation=attn_implementation
        )
        return pa.table([])

    def model_answer(self, query, temperature=None, max_new_tokens=None, format_answer_text=None,
                     save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                     replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """"""
        # intent recipie options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # code block
        temperature = temperature if isinstance(temperature, float) else 0.7
        max_new_tokens = max_new_tokens if isinstance(max_new_tokens, int) else 512
        format_answer_text = format_answer_text if isinstance(format_answer_text, bool) else True
        # set device
        device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        # Get the query answers from the embedding
        context_items = self.query_reranker(query=query, bi_limit=20, cross_limit=10)['text'].to_pylist()
        # Format the prompt with context items
        prompt = self._build_prompt(query=query, context_items=context_items)
        # Tokenize the prompt
        input_ids = self._tokenizer(prompt, return_tensors="pt").to(device)
        # Set tokenize terminators
        terminators = [
            self._tokenizer.eos_token_id,
            self._tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        # Generate an output of tokens
        outputs = self._model.generate(**input_ids,
                                 eos_token_id=terminators,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens,
                                 top_p=0.9,
                                 )
        # Turn the output tokens into text
        output_text = self._tokenizer.decode(outputs[0])
        if format_answer_text:
            # Replace special tokens and unnecessary help message
            output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace(
                "Sure, here is the answer to the user query:\n\n", "")
        question = pa.array([query], pa.string())
        context = pa.array([' '.join(context_items)], pa.string())
        answer = pa.array([output_text], pa.string())
        return pa.table([question, context, answer], names=['question', 'context', 'answer'])

    #  ---------
    #   Private
    #  ---------

    def _build_prompt(self, query: str, context_items: list[dict]) -> str:
        """"""
        # Join context items into one dotted paragraph
        context = "- " + "\n- ".join([item for item in context_items])
        # Create a base prompt with examples to help the model
        base_prompt = ("I am a professional summarizer with an academic audience. "
            "Based on the following context items, please answer the query. "
            "Give yourself room to think by extracting relevant passages from the context before answering the query. "
            "Don't return the thinking, only return the answer. Make sure your answers are as explanatory as possible. "
            "Now use the following context items to answer the user query:"
            "\n{context}"
            "\nUser query: {query}"
            "\nAnswer:")
        # Update base prompt with context items and query
        base_prompt = base_prompt.format(context=context, query=query)
        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user", "content": base_prompt}
        ]
        # Apply the chat template
        prompt = self._tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)
        return prompt

    def _template(self,
                  save_intent: bool=None, intent_level: [int, str]=None, intent_order: int=None,
                  replace_intent: bool=None, remove_duplicates: bool=None) -> pa.Table:
        """"""
        # intent recipie options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # code block

