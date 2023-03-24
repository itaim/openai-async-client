from abc import abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Generic, TypeVar

from pydantic import BaseModel
from pydantic.generics import GenericModel

from openai_async_client.reader import TextCompletionReader, ChatCompletionReader, CompletionReader


# see https://platform.openai.com/docs/api-reference/completions/create
# see https://platform.openai.com/docs/api-reference/chat/create

class OpenAIParams(BaseModel):
    model: str
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


DEFAULT_TEXT_PARAMS = OpenAIParams(
    model='text-davinci-003', temperature=0.5, max_tokens=1024, presence_penalty=1.0, frequency_penalty=2.0
)

DEFAULT_CHAT_PARAMS = OpenAIParams(
    model="gpt-3.5-turbo", temperature=0.5, max_tokens=None, presence_penalty=1.0, frequency_penalty=2.0
)


class Message(BaseModel):
    role: str
    content: str


class SystemMessage(Message):
    role = "system"


class EndpointConfig(Enum):
    TEXT = ("https://api.openai.com/v1/completions", TextCompletionReader())
    CHAT = ("https://api.openai.com/v1/chat/completions", ChatCompletionReader())

    def __init__(self, endpoint: str, reader: CompletionReader):
        self.endpoint = endpoint
        self.reader = reader


P = TypeVar("P")


class CompletionRequest(GenericModel, Generic[P]):
    key: Optional[Dict[str, Any]] = None
    params: Optional[OpenAIParams] = None
    prompt: P

    @abstractmethod
    def set_data(self, body):
        pass


class TextCompletionRequest(CompletionRequest[str]):
    def __init__(self, prompt: str, key: Optional[Dict[str, Any]] = None, params: Optional[OpenAIParams] = None):
        super().__init__(prompt=prompt, key=key, params=params or DEFAULT_TEXT_PARAMS)

    def set_data(self, body):
        body["prompt"] = self.prompt


class ChatCompletionRequest(CompletionRequest[List[Message]]):
    system: Optional[Message] = None

    def __init__(self, prompt: List[Message], system: Optional[Message] = None, key: Optional[Dict[str, Any]] = None,
                 params: Optional[OpenAIParams] = None):
        super().__init__(prompt=prompt, key=key, params=params or DEFAULT_CHAT_PARAMS)
        self.system = system

    def set_data(self, body):
        body["messages"] = ([] if not self.system else [self.system.dict()]) + [
            m.dict() for m in self.prompt
        ]
