import asyncio
import json
import logging
import os
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Union

import pandas as pd
from httpx import Timeout
from pandas import DataFrame
from pydantic import BaseModel

from async_requests import process_payloads, PostRequest, PostResult
from response import ResponseProcessor, DefaultChatResponseProcessor


# see https://platform.openai.com/docs/api-reference/chat/create


class ModelType(Enum):
    CHAT = "message"
    TEXT = "text"


class ModelConfig(BaseModel):
    type: ModelType
    encoding: str


MODELS_CONFIG = {
    "chat": ModelConfig(type=ModelType.CHAT, encoding="text-embedding-ada-002")
}


class OpenAIParams(BaseModel):
    model: str = "gpt-3.5-turbo"
    # set either temperature or top_p
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = None
    # defaults to inf
    max_tokens: Optional[int] = None
    # + increase prob of new topics
    presence_penalty: float = 0.0
    # + decrease prob of repetition
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[int, int]] = None
    user: Optional[str] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]]
    user: Optional[str]
    n: int = 1


DEFAULT_CHAT_PARAMS = OpenAIParams(
    temperature=0.5, max_tokens=None, presence_penalty=1.0, frequency_penalty=2.0
)


class Message(BaseModel):
    role: str
    content: str


class SystemMessage(Message):
    role = "system"


class ChatRequest(BaseModel):
    messages: List[Message]
    key: Optional[Dict[str, Any]] = None
    system: Optional[Message] = None
    params: Optional[OpenAIParams] = None


CHAT_MESSAGE_EXTRACTOR = DefaultChatResponseProcessor()


class OpenAIClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_chat_params: OpenAIParams = DEFAULT_CHAT_PARAMS,
    ):
        self._endpoint = f"https://api.openai.com/v1/chat/completions"
        api_key = api_key or os.environ["OPENAI_API_KEY"]
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self._default_chat_params = default_chat_params
        self._default_max_connections = 8
        self._default_max_retries = 5
        self._default_timeout = Timeout(3.0, read=20.0)

    @staticmethod
    def _from_json_response(
        result: PostResult, config: ModelConfig
    ) -> Union[str, List[str], BaseException]:
        if not result.value:
            return Exception("empty response")
        try:
            # json.loads(response)["choices"][0]["text"]
            choices = json.loads(result.value)["choices"]
            if config.type == ModelType.CHAT:
                return [choices[i]["message"]["content"] for i in range(len(choices))]
            else:
                return [choices[i]["text"] for i in range(len(choices))]
        except Exception as e:
            logging.exception(f"choices extraction {e}")
            return e

    @staticmethod
    def create_openai_body(params: OpenAIParams) -> Dict[str, Any]:
        open_ai_body = {
            "model": params.model,
            "n": params.n,
            "presence_penalty": params.presence_penalty,
            "frequency_penalty": params.frequency_penalty,
        }

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

    def chat_completion(
        self,
        request: ChatRequest,
        return_raw_response: bool = False,
        response_processor: ResponseProcessor = CHAT_MESSAGE_EXTRACTOR,
    ) -> Union[str, List[str]]:
        body = self.create_openai_body(request.params or self._default_chat_params)
        body["messages"] = [m.dict() for m in request.messages]
        if request.key:
            key = request.key
            body["user"] = ":".join([f"{k}-{v}" for k, v in key.items()])
        else:
            key = {"key": "chat"}
        response = asyncio.run(
            process_payloads(
                endpoint=self._endpoint,
                headers=self._headers,
                requests=[PostRequest(key=key, body=body)],
                max_concurrent_connections=self._default_max_connections,
                max_retries=self._default_max_retries,
            )
        )[0]
        if response.error:
            raise response.error
        elif not return_raw_response and response_processor:
            result = response_processor(response.value)
            if isinstance(result, BaseException):
                raise result
            return result
        else:
            return response.value

    def chat_completions(
        self,
        df: DataFrame,
        request_fn: Callable[[pd.Series], ChatRequest],
        response_col: str = "openai_reply",
        max_connections: Optional[int] = None,
        max_retries: Optional[int] = None,
        return_raw_response: bool = False,
        response_processor: ResponseProcessor = CHAT_MESSAGE_EXTRACTOR,
    ) -> Optional[DataFrame]:
        if len(df.index) == 0:
            logging.error(f"Empty input")
            return None

        def to_request(r: pd.Series) -> PostRequest:
            request = request_fn(r)
            body = self.create_openai_body(request.params or self._default_chat_params)
            key = request.key
            messages = ([] if not request.system else [request.system.dict()]) + [
                m.dict() for m in request.messages
            ]
            body["messages"] = messages
            body["user"] = ":".join([f"{k}-{v}" for k, v in key.items()])
            return PostRequest(key=key, body=body)

        payloads = df.apply(lambda r: to_request(r), axis=1).values.tolist()
        responses = asyncio.run(
            process_payloads(
                endpoint=self._endpoint,
                headers=self._headers,
                requests=payloads,
                max_concurrent_connections=max_connections
                or self._default_max_connections,
                max_retries=max_retries or self._default_max_retries,
            )
        )

        def to_record(
            response: Union[PostResult, BaseException]
        ) -> Optional[Dict[str, Any]]:
            if isinstance(response, BaseException):
                return None
            record = response.key.copy()
            if response.error:
                record[response_col] = pd.NA
                record["api_error"] = response.error.__repr__()

            elif not return_raw_response and response_processor:
                processed = response_processor(response.value)
                if isinstance(processed, BaseException):
                    record[response_col] = pd.NA
                    record["api_error"] = response.error.__repr__()
                else:
                    record[response_col] = processed
                    record["api_error"] = pd.NA
            return record

        records = filter(lambda x: x, [to_record(r) for r in responses])
        key_cols = list(payloads[0].key.keys())
        res_df = pd.DataFrame.from_records(data=records)
        combined = df.merge(res_df, on=key_cols)
        return combined
