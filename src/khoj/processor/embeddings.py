import logging
import os
from typing import List
from urllib.parse import urlparse

import openai
import requests
import tqdm
from sentence_transformers import CrossEncoder, SentenceTransformer
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from torch import nn

from khoj.database.models import SearchModelConfig
from khoj.utils.helpers import (
    fix_json_dict,
    get_device,
    get_openai_client,
    merge_dicts,
    timer,
)
from khoj.utils.rawconfig import SearchResponse

logger = logging.getLogger(__name__)
DEFAULT_EMBEDDINGS_BATCH_SIZE = 1000
EMBEDDINGS_BATCH_SIZE_ENV_VAR = "KHOJ_EMBEDDINGS_BATCH_SIZE"


class EmbeddingsModel:
    def __init__(
        self,
        model_name: str = "thenlper/gte-small",
        embeddings_inference_endpoint: str = None,
        embeddings_inference_endpoint_api_key: str = None,
        embeddings_inference_endpoint_type=SearchModelConfig.ApiType.LOCAL,
        embeddings_batch_size: int = None,
        query_encode_kwargs: dict = {},
        docs_encode_kwargs: dict = {},
        model_kwargs: dict = {},
    ):
        default_query_encode_kwargs = {"show_progress_bar": False, "normalize_embeddings": True}
        default_docs_encode_kwargs = {"show_progress_bar": True, "normalize_embeddings": True}
        self.query_encode_kwargs = merge_dicts(fix_json_dict(query_encode_kwargs), default_query_encode_kwargs)
        self.docs_encode_kwargs = merge_dicts(fix_json_dict(docs_encode_kwargs), default_docs_encode_kwargs)
        self.model_kwargs = merge_dicts(fix_json_dict(model_kwargs), {"device": get_device()})
        self.model_name = model_name
        self.inference_endpoint = embeddings_inference_endpoint
        self.api_key = embeddings_inference_endpoint_api_key
        self.inference_endpoint_type = embeddings_inference_endpoint_type
        self.embedding_batch_size = self.resolve_embedding_batch_size(embeddings_batch_size)
        if self.inference_endpoint_type == SearchModelConfig.ApiType.LOCAL:
            with timer(f"Loaded embedding model {self.model_name}", logger):
                self.embeddings_model = SentenceTransformer(self.model_name, **self.model_kwargs)

    @staticmethod
    def parse_embedding_batch_size(value, source: str) -> int:
        if value is None:
            return None
        try:
            parsed_value = int(value)
        except (TypeError, ValueError):
            logger.warning(
                f"Ignoring invalid embedding batch size from {source}: {value}. "
                f"Using default {DEFAULT_EMBEDDINGS_BATCH_SIZE}."
            )
            return None
        if parsed_value <= 0:
            logger.warning(
                f"Ignoring non-positive embedding batch size from {source}: {value}. "
                f"Using default {DEFAULT_EMBEDDINGS_BATCH_SIZE}."
            )
            return None
        return parsed_value

    def resolve_embedding_batch_size(self, configured_batch_size: int) -> int:
        model_batch_size = self.parse_embedding_batch_size(
            configured_batch_size,
            "SearchModelConfig.embeddings_batch_size",
        )
        if model_batch_size is not None:
            return model_batch_size
        env_batch_size = self.parse_embedding_batch_size(
            os.getenv(EMBEDDINGS_BATCH_SIZE_ENV_VAR),
            EMBEDDINGS_BATCH_SIZE_ENV_VAR,
        )
        if env_batch_size is not None:
            return env_batch_size
        return DEFAULT_EMBEDDINGS_BATCH_SIZE

    def embed_query(self, query):
        if self.inference_endpoint_type == SearchModelConfig.ApiType.HUGGINGFACE:
            return self.embed_with_hf([query])[0]
        elif self.inference_endpoint_type == SearchModelConfig.ApiType.OPENAI:
            return self.embed_with_openai([query])[0]
        return self.embeddings_model.encode([query], **self.query_encode_kwargs)[0]

    @retry(
        retry=retry_if_exception_type(requests.exceptions.HTTPError),
        wait=wait_random_exponential(multiplier=1, max=10),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
    )
    def embed_with_hf(self, docs):
        payload = {"inputs": docs}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(self.inference_endpoint, json=payload, headers=headers)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(
                f" Error while calling inference endpoint {self.inference_endpoint} with error {e}, response {response.json()} ",
                exc_info=True,
            )
            raise e
        return response.json()["embeddings"]

    @retry(
        retry=retry_if_exception_type(requests.exceptions.HTTPError),
        wait=wait_random_exponential(multiplier=1, max=10),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
    )
    def embed_with_openai(self, docs):
        client = get_openai_client(self.api_key, self.inference_endpoint)
        response = client.embeddings.create(input=docs, model=self.model_name, encoding_format="float")
        return [item.embedding for item in response.data]

    def embed_documents(self, docs):
        if self.inference_endpoint_type == SearchModelConfig.ApiType.LOCAL:
            return self.embeddings_model.encode(docs, **self.docs_encode_kwargs).tolist() if docs else []
        elif self.inference_endpoint_type == SearchModelConfig.ApiType.HUGGINGFACE:
            embed_with_api = self.embed_with_hf
        elif self.inference_endpoint_type == SearchModelConfig.ApiType.OPENAI:
            embed_with_api = self.embed_with_openai
        else:
            logger.warning(
                f"Unsupported inference endpoint: {self.inference_endpoint_type}. Generating embeddings locally instead."
            )
            return self.embeddings_model.encode(docs, **self.docs_encode_kwargs).tolist()
        # Break up docs payload into smaller chunks for compatibility with API batch limits.
        embeddings = []
        with tqdm.tqdm(total=len(docs)) as pbar:
            for i in range(0, len(docs), self.embedding_batch_size):
                docs_to_embed = docs[i : i + self.embedding_batch_size]
                generated_embeddings = embed_with_api(docs_to_embed)
                embeddings += generated_embeddings
                pbar.update(len(docs_to_embed))
        return embeddings


class CrossEncoderModel:
    def __init__(
        self,
        model_name: str = "mixedbread-ai/mxbai-rerank-xsmall-v1",
        cross_encoder_inference_endpoint: str = None,
        cross_encoder_inference_endpoint_api_key: str = None,
        model_kwargs: dict = {},
    ):
        self.model_name = model_name
        self.inference_endpoint = cross_encoder_inference_endpoint
        self.api_key = cross_encoder_inference_endpoint_api_key
        self.model_kwargs = merge_dicts(model_kwargs, {"device": get_device()})
        with timer(f"Loaded cross-encoder model {self.model_name}", logger):
            self.cross_encoder_model = CrossEncoder(model_name=self.model_name, **self.model_kwargs)

    def inference_server_enabled(self) -> bool:
        return self.api_key is not None and self.inference_endpoint is not None

    def predict(self, query, hits: List[SearchResponse], key: str = "compiled"):
        if self.inference_server_enabled() and "huggingface" in self.inference_endpoint:
            target_url = f"{self.inference_endpoint}"
            payload = {"inputs": {"query": query, "passages": [hit.additional[key] for hit in hits]}}
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            response = requests.post(target_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["scores"]

        cross_inp = [[query, hit.additional[key]] for hit in hits]
        cross_scores = self.cross_encoder_model.predict(cross_inp, activation_fct=nn.Sigmoid())
        return cross_scores
