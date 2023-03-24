import uuid
from typing import Callable

import pandas as pd
from dotenv import load_dotenv

from openai_async_client import AsyncCreate, Message, ChatCompletionRequest, SystemMessage, OpenAIParams
from openai_async_client.model import TextCompletionRequest, EndpointConfig
from openai_async_client.openai import EMPTY_RECORD

load_dotenv()

TEST_INPUTS = [
    "the open society and its enemies by Karl Popper",
    "Das Capital by Karl Marx",
    "Pride and Prejudice by Jane Austen",
    "Frankenstein by Mary Shelley",
    "Moby Dick by  Herman Melville",
]


def test_chat_completion():
    create = AsyncCreate()
    messages = [
        Message(
            role="user",
            content=f"Hello ChatGPT, Give a brief overview of the book {TEST_INPUTS[3]}.",
        )
    ]
    response = create.completion(ChatCompletionRequest(prompt=messages))
    print(f"\n{response}")
    assert len(response.text) > 100


def test_text_completion():
    create = AsyncCreate()
    response = create.completion(
        TextCompletionRequest(prompt=f"Hello ChatGPT, Give a brief overview of the book {TEST_INPUTS[1]}."))
    print(f"\n{response}")
    assert len(response.text) > 100


def my_chat_prompt_fn(n: int):
    def chat_prompt_fn(r: pd.Series) -> ChatCompletionRequest:
        message = Message(
            role="user",
            content=f"Hello ChatGPT, Give a brief overview of the book {r.book_name}.",
        )
        key = {"user_id": r.user_id, "book_id": r.book_id}
        return ChatCompletionRequest(
            key=key,
            prompt=[message],
            system=SystemMessage(content="Assistant is providing book reviews"),
            params=OpenAIParams(model="gpt-3.5-turbo", n=n)
        )

    return chat_prompt_fn


def do_completions(prompt_fn: Callable, config: EndpointConfig):
    records = [
        {"user_id": i, "book_id": str(uuid.uuid4())[:6], "book_name": s}
        for i, s in enumerate(TEST_INPUTS)
    ]
    input_df = pd.DataFrame.from_records(records)
    client = AsyncCreate()

    res_df = client.completions(df=input_df, request_fn=prompt_fn, config=config)

    print(res_df.columns)
    assert len(res_df.index) == len(input_df.index)
    assert set(res_df.columns) == set(input_df.columns).union(EMPTY_RECORD)
    return res_df


def test_chat_completions_single_choice():
    n = 1
    res_df = do_completions(my_chat_prompt_fn(n), EndpointConfig.CHAT)
    res_df.to_csv(f'data/chat_completions_{n}.csv', index=False)


def test_chat_completions_multi_choice():
    n = 3
    res_df = do_completions(my_chat_prompt_fn(n), EndpointConfig.CHAT)
    res_df.to_csv(f'data/chat_completions_{n}.csv', index=False)


def my_text_prompt_fn(n: int):
    def text_prompt_fn(r: pd.Series) -> TextCompletionRequest:
        key = {"user_id": r.user_id, "book_id": r.book_id}
        return TextCompletionRequest(
            key=key,
            prompt=f"Hello ChatGPT, Give a brief overview of the book {r.book_name}.",
            params=OpenAIParams(model='text-davinci-003', n=n)
        )

    return text_prompt_fn


def test_text_completions_single_choice():
    n = 1
    res_df = do_completions(my_text_prompt_fn(n), EndpointConfig.CHAT)
    res_df.to_csv(f'data/text_completions_{n}.csv', index=False)


def test_text_completions_multi_choice():
    n = 3
    res_df = do_completions(my_text_prompt_fn(n), EndpointConfig.CHAT)
    res_df.to_csv(f'data/text_completions_{n}.csv', index=False)
