import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar, Generic, Union, Dict, List

from pydantic import BaseModel


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Completion(BaseModel):
    id: str
    created: int
    text: Union[str, List[str]]
    usage: Usage


R = TypeVar("R",bound=Completion)
C = TypeVar("C")


class CompletionReader(Generic[R, C], Callable[..., R], ABC):

    @staticmethod
    def _read_usage(usage: Dict[str, int]) -> Usage:
        return Usage(prompt_tokens=usage['prompt_tokens'], completion_tokens=usage['completion_tokens'],
                     total_tokens=usage['total_tokens'])

    @abstractmethod
    def _read_choice(self, choice: Dict[str, C]) -> str:
        pass

    @abstractmethod
    def __call__(self, json: str, *args: Any, **kwargs: Any) -> R:
        pass


class BaseCompletionReader(CompletionReader[Completion, C], Callable[..., Completion], ABC):

    @abstractmethod
    def _read_choice(self, choice: Dict[str, C]) -> str:
        pass

    def __call__(
            self, body: str, *args: Any, **kwargs: Any
    ) -> Union[Completion, BaseException]:
        if not body:
            return Exception("empty response")
        try:
            json_obj = json.loads(body)
            usage = self._read_usage(json_obj['usage'])
            choices = json_obj["choices"]
            if len(choices) == 1:
                text = self._read_choice(choices[0])
            else:
                text = [self._read_choice(choices[i]) for i in range(len(choices))]
            return Completion(id=json_obj['id'], created=json_obj['created'], text=text, usage=usage)
        except Exception as e:
            logging.exception(f"choices extraction {e}")
            return e


class TextCompletionReader(BaseCompletionReader[str]):

    def _read_choice(self, choice: Dict[str, str]) -> str:
        return choice['text']


class ChatCompletionReader(BaseCompletionReader[Dict[str, str]]):

    def _read_choice(self, choice: Dict[str, Dict[str, str]]) -> str:
        return choice['message']['content']
