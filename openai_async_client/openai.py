import asyncio
import logging
import os
from typing import Dict, Any, Optional, Callable, Union

import pandas as pd
from httpx import Timeout
from pandas import DataFrame

from openai_async_client.async_requests import process_payloads, PostRequest, PostResult
from openai_async_client.model import EndpointConfig, TextCompletionRequest
from openai_async_client.model import OpenAIParams, CompletionRequest, ChatCompletionRequest
from openai_async_client.reader import Completion

DEFAULT_TIMEOUT = Timeout(3.0, read=20.0)
DEFAULT_RETRIES = 5
DEFAULT_MAX_CONNECTIONS = 8

EMPTY_RECORD = {
    "openai_id": pd.NA,
    "openai_created": pd.NA,
    "openai_completion": pd.NA,
    "openai_prompt_tokens": pd.NA,
    "openai_completion_tokens": pd.NA,
    "openai_total_tokens": pd.NA,
    "api_error": pd.NA,
}


class AsyncCreate:
    def __init__(
            self,
            api_key: Optional[str] = None,
    ):
        api_key = api_key or os.environ["OPENAI_API_KEY"]
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    @staticmethod
    def get_endpoint_config(request) -> EndpointConfig:
        if isinstance(request, ChatCompletionRequest):
            return EndpointConfig.CHAT
        elif isinstance(request, TextCompletionRequest):
            return EndpointConfig.TEXT
        else:
            raise Exception(f'unknown request {request}')

    @staticmethod
    def create_openai_body(params: OpenAIParams) -> Dict[str, Any]:
        open_ai_body = {
            "model": params.model,
            "presence_penalty": params.presence_penalty,
            "frequency_penalty": params.frequency_penalty,
        }
        if params.n > 1:
            open_ai_body["n"] = params.n
        if params.max_tokens:
            open_ai_body["max_tokens"] = params.max_tokens
        if params.stop:
            open_ai_body["stop"] = params.stop
        if params.logit_bias:
            open_ai_body["logit_bias"] = params.logit_bias
        if params.temperature:
            open_ai_body["temperature"] = params.temperature
        elif params.top_p:
            open_ai_body["top_p"] = params.top_p
        if params.temperature and params.top_p:
            logging.warning(
                f"both temperature and top_p were set, using temperature {params.temperature}"
            )
        return open_ai_body

    def completion(
            self,
            request: CompletionRequest,
            client_timeout: Timeout = DEFAULT_TIMEOUT,
            retries: int = DEFAULT_RETRIES,
            return_raw_response: bool = False
    ) -> Completion:
        body = self.create_openai_body(request.params)
        request.set_data(body)
        if request.key:
            key = request.key
            body["user"] = ":".join([f"{k}-{v}" for k, v in key.items()])
        else:
            key = {"key": "completion"}
        config = self.get_endpoint_config(request)
        response = asyncio.run(
            process_payloads(
                endpoint=config.endpoint,
                headers=self._headers,
                requests=[PostRequest(key=key, body=body)],
                max_concurrent_connections=1,
                max_retries=retries,
                timeout=client_timeout,
            )
        )[0]
        if response.error:
            raise response.error
        elif return_raw_response:
            return response.value
        else:
            result = config.reader(response.value)
            if isinstance(result, BaseException):
                raise result
            return result

    def completions(
            self,
            df: DataFrame,
            request_fn: Callable[[pd.Series], CompletionRequest],
            config: EndpointConfig,
            max_connections: int = DEFAULT_MAX_CONNECTIONS,
            max_retries: int = DEFAULT_RETRIES,
            client_timeout: Timeout = DEFAULT_TIMEOUT,
            return_raw_response: bool = False
    ) -> Optional[DataFrame]:
        if len(df.index) == 0:
            logging.error(f"Empty input")
            return None

        def to_request(r: pd.Series) -> PostRequest:
            request = request_fn(r)
            body = self.create_openai_body(request.params)
            request.set_data(body)
            body["user"] = ":".join([f"{k}-{v}" for k, v in request.key.items()])
            return PostRequest(key=request.key, body=body)

        payloads = df.apply(lambda r: to_request(r), axis=1).values.tolist()
        responses = asyncio.run(
            process_payloads(
                endpoint=config.endpoint,
                headers=self._headers,
                requests=payloads,
                timeout=client_timeout,
                max_concurrent_connections=max_connections,
                max_retries=max_retries,
            )
        )

        def to_record(
                response: Union[PostResult, BaseException]
        ) -> Optional[Dict[str, Any]]:
            if isinstance(response, BaseException):
                return None
            record = response.key.copy()
            if response.error:
                if not return_raw_response:
                    record.update(EMPTY_RECORD)
                record["api_error"] = response.error.__repr__()
            elif return_raw_response:
                record['openai_completion'] = response.value
                record['api_error'] = pd.NA
            else:
                completion = config.reader(response.value)
                if isinstance(completion, BaseException):
                    record.update(EMPTY_RECORD)
                    record["api_error"] = response.error.__repr__()
                else:
                    record["openai_id"] = completion.id
                    record["openai_created"] = completion.created
                    record["openai_completion"] = completion.text
                    record["openai_prompt_tokens"] = completion.usage.prompt_tokens
                    record["openai_completion_tokens"] = completion.usage.completion_tokens
                    record["openai_total_tokens"] = completion.usage.total_tokens
                    record["api_error"] = pd.NA
            return record

        records = filter(lambda x: x, [to_record(r) for r in responses])
        key_cols = list(payloads[0].key.keys())
        res_df = pd.DataFrame.from_records(data=records)
        combined = df.merge(res_df, on=key_cols)
        return combined
