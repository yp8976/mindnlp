# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""table question answering pipeline."""
import collections
import types

from .base import ArgumentHandler, Dataset, Pipeline, PipelineException

from ...utils import (
    is_mindspore_available,
)

if is_mindspore_available():
    import mindspore
    import mindspore.nn
    from mindspore import ops
    from ..models.auto.modeling_auto import (
        MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
    )


class TableQuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    Handles arguments for the TableQuestionAnsweringPipeline
    """

    def __call__(self, table=None, query=None, **kwargs):
        """
        Returns tqa_pipeline_inputs of shape:
        [
            {"table": pd.DataFrame, "query": List[str]},
            ...,
            {"table": pd.DataFrame, "query" : List[str]}
        ]
        Args:
            table:
            query:

        Returns:

        """
        import pandas as pd

        if table is None:
            raise ValueError("Keyword argument `table` cannot be None.")
        elif query is None:
            if isinstance(table, dict) and table.get("table") is not None and table.get("query") is not None:
                tqa_pipeline_inputs = [table]
            elif isinstance(table, list) and len(table) > 0:
                if not all(isinstance(d, dict) for d in table):
                    raise ValueError(
                        f"Keyword argument `table` should be a list of dict, but is {(type(d) for d in table)}"
                    )

                if table[0].get("query") is not None and table[0].get("table") is not None:
                    tqa_pipeline_inputs = table
                else:
                    raise ValueError(
                        "If keyword argument `table` is a list of dictionaries, each dictionary should have a `table`"
                        f" and `query` key, but only dictionary has keys {table[0].keys()} `table` and `query` keys."
                    )
            elif Dataset is not None and isinstance(table, Dataset) or isinstance(table, types.GeneratorType):
                return table
            else:
                raise ValueError(
                    "Invalid input. Keyword argument `table` should be either of type `dict` or `list`, but "
                    f"is {type(table)})"
                )
        else:
            tqa_pipeline_inputs = [{"table": table, "query": query}]

        for tqa_pipeline_input in tqa_pipeline_inputs:
            if not isinstance(tqa_pipeline_input["table"], pd.DataFrame):
                if tqa_pipeline_input["table"] is None:
                    raise ValueError("Table cannot be None.")

                tqa_pipeline_input["table"] = pd.DataFrame(tqa_pipeline_input["table"])

        return tqa_pipeline_inputs


class TableQuestionAnsweringPipeline(Pipeline):
    """
    Table Question Answering pipeline using a `ModelForTableQuestionAnswering`. This pipeline is only available in
    PyTorch.

    Example:

    ```python
    >>> from mindnlp.transformers import pipeline

    >>> oracle = pipeline(model="google/tapas-base-finetuned-wtq")
    >>> table = {
    ...     "Repository": ["Transformers", "Datasets", "Tokenizers"],
    ...     "Stars": ["36542", "4512", "3934"],
    ...     "Contributors": ["651", "77", "34"],
    ...     "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
    ... }
    >>> oracle(query="How many stars does the transformers repository have?", table=table)
    {'answer': 'AVERAGE > 36542', 'coordinates': [(0, 1)], 'cells': ['36542'], 'aggregator': 'AVERAGE'}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This tabular question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"table-question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a tabular question answering task.
    See the up-to-date list of available models on
    [hf-mirror.com/models](https://hf-mirror.com/models?filter=table-question-answering).
    """

    def __init__(self, *args, args_paser=TableQuestionAnsweringArgumentHandler(), **kwargs):
        super().__init__(*args, **kwargs)
        self._args_paser = args_paser

        mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES.copy()
        mapping.update(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES)
        self.check_model_type(mapping)

        self.aggregate = bool(getattr(self.model.config, "aggregation_labels", False)) and bool(
            getattr(self.model.config, "aggregation_labels", False)
        )
        self.type = "tapas" if hasattr(self.model.config, "aggregation_labels") else None

    def __call__(self, *args, **kwargs):
        r"""
        Answers queries according to a table. The pipeline accepts several types of inputs which are detailed below:

        - `pipeline(table, query)`
        - `pipeline(table, [query])`
        - `pipeline(table=table, query=query)`
        - `pipeline(table=table, query=[query])`
        - `pipeline({"table": table, "query": query})`
        - `pipeline({"table": table, "query": [query]})`
        - `pipeline([{"table": table, "query": query}, {"table": table, "query": query}])`

        The `table` argument should be a dict or a DataFrame built from that dict, containing the whole table:

        Example:

        ```python
        data = {
            "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
            "age": ["56", "45", "59"],
            "number of movies": ["87", "53", "69"],
            "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
        }
        ```

        This dictionary can be passed in as such, or can be converted to a pandas DataFrame:

        Example:

        ```python
        import pandas as pd

        table = pd.DataFrame.from_dict(data)
        ```

        Args:
            table (`pd.DataFrame` or `Dict`):
                Pandas DataFrame or dictionary that will be converted to a DataFrame containing all the table values.
                See above for an example of dictionary.
            query (`str` or `List[str]`):
                Query or list of queries that will be sent to the model alongside the table.
            sequential (`bool`, *optional*, defaults to `False`):
                Whether to do inference sequentially or as a batch. Batching is faster, but models like SQA require the
                inference to be done sequentially to extract relations within sequences, given their conversational
                nature.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).

            truncation (`bool`, `str` or [`TapasTruncationStrategy`], *optional*, defaults to `False`):
                Activates and controls truncation. Accepts the following values:

                - `True` or `'drop_rows_to_fit'`: Truncate to a maximum length specified with the argument `max_length`
                  or to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate row by row, removing rows from the table.
                - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).


        Return:
            A dictionary or a list of dictionaries containing results: Each result is a dictionary with the following
            keys:

            - **answer** (`str`) -- The answer of the query given the table. If there is an aggregator, the answer will
              be preceded by `AGGREGATOR >`.
            - **coordinates** (`List[Tuple[int, int]]`) -- Coordinates of the cells of the answers.
            - **cells** (`List[str]`) -- List of strings made up of the answer cell values.
            - **aggregator** (`str`) -- If the model has an aggregator, this returns the aggregator.
        """
        tqa_pipline_inputs = self._args_paser(*args, **kwargs)

        return super().__call__(tqa_pipline_inputs, **kwargs)

    def _sanitize_parameters(self, sequential=None, padding=None, truncation=None, **kwargs):
        preprocess_param = {}
        if padding is not None:
            preprocess_param["padding"] = padding
        if truncation is not None:
            preprocess_param["truncation"] = truncation

        forword_param = {}
        if sequential is not None:
            forword_param["sequential"] = sequential

        return preprocess_param, forword_param, {}

    def preprocess(self, pipline_input, sequential=None, padding=True, truncation=None):
        if truncation is None:
            truncation = "drop_rows_to_fit" if self.type == "tapas" \
                else "do_not_truncate"

        table, query = pipline_input["table"], pipline_input["query"]
        if table.empty:
            raise ValueError("table is empty")
        if query is None or query == "":
            raise ValueError("query is empty")

        inputs = self.tokenizer(
            table,
            query,
            truncation=truncation,
            padding=padding,
            return_tensors="ms"
        )
        inputs["table"] = table
        return inputs

    def _forward(self, model_inputs, sequential=False, **kwargs):
        table = model_inputs.pop("table")

        if self.type == "tapas":
            if sequential:
                outputs = self.sequential_inference(**model_inputs)
            else:
                outputs = self.batch_inference(**model_inputs)
        else:
            outputs = self.model.generate(**model_inputs, **kwargs)

        model_outputs = {"model_inputs": model_inputs, "table": table, "outputs": outputs}
        return model_outputs

    def postprocess(self, model_outputs, **postprocess_parameters):
        inputs = model_outputs["model_inputs"]
        table = model_outputs["table"]
        outputs = model_outputs["outputs"]
        if self.type == "tapas":
            if self.aggregate:
                logits, logits_agg = outputs[:2]
                predictions = self.tokenizer.convert_logits_to_predictions(inputs, logits, logits_agg)
                answer_coordinates_batch, agg_predictions = predictions
                aggregators = {i: self.model.config.aggregation_labels[pred] for i, pred in enumerate(agg_predictions)}

                no_agg_label_index = self.model.config.no_aggregation_label_index
                aggregators_prefix = {
                    i: aggregators[i] + " > " for i, pred in enumerate(agg_predictions) if pred != no_agg_label_index
                }
            else:
                logits = outputs[0]
                predictions = self.tokenizer.convert_logits_to_predictions(inputs, logits)
                answer_coordinates_batch = predictions[0]
                aggregators = {}
                aggregators_prefix = {}
            answers = []
            for index, coordinates in enumerate(answer_coordinates_batch):
                cells = [table.iat[coordinate] for coordinate in coordinates]
                aggregator = aggregators.get(index, "")
                aggregator_prefix = aggregators_prefix.get(index, "")
                answer = {
                    "answer": aggregator_prefix + ", ".join(cells),
                    "coordinates": coordinates,
                    "cells": [table.iat[coordinate] for coordinate in coordinates],
                }
                if aggregator:
                    answer["aggregator"] = aggregator

                answers.append(answer)
            if len(answer) == 0:
                raise PipelineException("Empty answer", self.model.model_tags, "Answer is empty!")
        else:
            answers = [{"answer": answer} for answer in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]

        return answers if len(answers) > 1 else answers[0]

    def batch_inference(self, **inputs):
        return self.model(**inputs)

    def sequential_inference(self, **inputs):
        """
        Inference used for models that need to process sequences in a sequential fashion, like the SQA models which
        handle conversational query related to a table.
        """
        all_logits = []
        all_aggregations = []
        prev_answers = None
        batch_size = inputs["input_ids"].shape[0]

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        token_type_ids_example = None

        for index in range(batch_size):
            # If sequences have already been processed, the token type IDs will be created according to the previous
            # answer.
            if prev_answers is not None:
                prev_labels_example = token_type_ids_example[:, 3]  # shape(seq_len,)
                model_labels = mindspore.numpy.zeros_like(  # shape(seq_len,)
                    prev_labels_example,
                    dtype=mindspore.dtype.int32
                )

                token_type_ids_example = token_type_ids[index]
                for i in range(model_labels.shape[0]):
                    segment_id = token_type_ids_example[:, 0].tolist()[i]
                    col = token_type_ids_example[:, 1].tolist()[i] - 1
                    row = token_type_ids_example[:, 2].tolist()[i] - 1

                    if col >= 0 and row >= 0 and segment_id == 1:
                        model_labels[i] = int(prev_answers[(col, row)])

                token_type_ids_example[:, 3] = mindspore.Tensor(model_labels).type(mindspore.int64)

            input_ids_example = input_ids[index]
            attention_mask_example = attention_mask[index]
            token_type_ids_example = token_type_ids[index]
            outputs = self.model(
                input_ids=input_ids_example.unsqueeze(0),
                attention_mask=attention_mask_example.unsqueeze(0),
                token_type_ids=token_type_ids_example.unsqueeze(0),
            )
            logits = outputs.logits

            if self.aggregate:
                all_aggregations.append(outputs.logis_aggregation)

            all_logits.append(logits)

            sigmoid = ops.Sigmoid()
            probs = sigmoid(logits)
            epsilon = 1e-6
            probs = ops.clip_by_value(probs, epsilon, 1 - epsilon)
            dist_per_token = mindspore.nn.probability.distribution.Bernoulli(probs=probs)
            probabilities = dist_per_token.probs * attention_mask_example.type(mindspore.dtype.float32)

            coords_to_probs = collections.defaultdict(list)
            for i, p in enumerate(probabilities.squeeze(0).tolist()):
                segment_id = token_type_ids_example[:, 0].tolist()[i]
                col = token_type_ids_example[:, 1].tolist()[i] - 1
                row = token_type_ids_example[:, 2].tolist()[i] - 1
                if col >= 0 and row >= 0 and segment_id == 1:
                    coords_to_probs[(col, row)].append(p)

            prev_answers = {key: mindspore.numpy.array(coords_to_probs[key]).mean() > 0.5
                            for key in coords_to_probs}

        logits_batch = mindspore.ops.cat(tuple(all_logits), axis=0)

        return (logits_batch,) if not self.aggregate \
            else (logits_batch, mindspore.ops.cat(tuple(all_aggregations), axis=0))
