import uuid

import pandas as pd
from dotenv import load_dotenv

from rolebotics.openai_client import OpenAIClient, ChatRequest, Message, SystemMessage

load_dotenv()

TEST_INPUTS = [
    "the open society and its enemies by Karl Popper",
    "Das Capital by Karl Marx",
    "Pride and Prejudice by Jane Austen",
    "Frankenstein by Mary Shelley",
    "Moby Dick by  Herman Melville",
]


def test_chat_completion():
    client = OpenAIClient()
    messages = [
        Message(
            role="user",
            content=f"Hello ChatGPT, Give a brief overview of the book {TEST_INPUTS[3]}.",
        )
    ]
    response = client.chat_completion(ChatRequest(messages=messages))
    print(f"\n{response}")
    assert len(response) > 100


def test_chat_completions():
    records = [
        {"user_id": i, "book_id": str(uuid.uuid4())[:6], "book_name": s}
        for i, s in enumerate(TEST_INPUTS)
    ]
    input_df = pd.DataFrame.from_records(records)
    client = OpenAIClient()

    def request_fn(r: pd.Series) -> ChatRequest:
        message = Message(
            role="user",
            content=f"Hello ChatGPT, Give a brief overview of the book {r.book_name}.",
        )
        key = {"user_id": r.user_id, "book_id": r.book_id}
        return ChatRequest(
            key=key,
            messages=[message],
            system=SystemMessage(content="Assistant is providing book reviews"),
        )

    res_df = client.chat_completions(input_df, request_fn)
    print(res_df.columns)
    assert len(res_df.index) == len(input_df.index)
    assert set(res_df.columns) == set(input_df.columns).union(
        {"openai_reply", "api_error"}
    )
    # res_df.to_csv(f"data/test_res.csv", index=False)
