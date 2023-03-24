import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar, Generic, Union, List

# Define a type variable for the result of the callable
R = TypeVar("R")


class ResponseProcessor(Generic[R], Callable[..., R], ABC):
    @abstractmethod
    def __call__(self, json: str, *args: Any, **kwargs: Any) -> R:
        pass


class DefaultChatResponseProcessor(ResponseProcessor[str]):
    def __call__(
            self, body: str, *args: Any, **kwargs: Any
    ) -> Union[str, List[str], BaseException]:
        if not body:
            return Exception("empty response")
        try:
            choices = json.loads(body)["choices"]
            if len(choices) == 1:
                return choices[0]["message"]["content"]
            else:
                return [choices[1]["message"]["content"] for i in range(len(choices))]
        except Exception as e:
            logging.exception(f"choices extraction {e}")
            return e
