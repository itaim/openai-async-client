# OpenAI client with client timeout and parallel processing

## Quick Install

`pip install openai-async-client`

## 🤔 What is this?

This library is aimed at assisting with OpenAI API usage by:

Support for client side timeouts with retry and backoff for completions.

Support for concurrent processing with pandas DataFrames.

##

Example of chat completion with client timeout of 1 second to connect and 10 seconds to read with a maximum of 3
retries.

````
import os
from httpx import Timeout
from openai_async_client import OpenAIAsync, ChatRequest, Message, SystemMessage

client = OpenAIAsync(api_key=os.environ['OPENAI_API_KEY'])

messages = [
    Message(
        role="user",
        content=f"Hello ChatGPT, Give a brief overview of the book Frankenstein by Mary Shelley.",
    )
]

response = client.chat_completion(request=ChatRequest(messages=messages),client_timeout=Timeout(1.0,read=10.0),retries=3)

````

Example of concurrent processing a DataFrame for chat completions with 4 concurrent connections.

 ````
import os
from httpx import Timeout
from openai_async_client import OpenAIAsync, ChatRequest, Message, SystemMessage
import uuid
import pandas as pd

[//]: # (Example DataFrame)
TEST_INPUTS = [
    "the open society and its enemies by Karl Popper",
    "Das Capital by Karl Marx",
    "Pride and Prejudice by Jane Austen",
    "Frankenstein by Mary Shelley",
    "Moby Dick by  Herman Melville",
]

records = [
    {"user_id": i, "book_id": str(uuid.uuid4())[:6], "book_name": s}
    for i, s in enumerate(TEST_INPUTS)
]
input_df = pd.DataFrame.from_records(records)


client = OpenAIAsync(api_key=os.environ['OPENAI_API_KEY'])

[//]: # (Define a mapping function from row to prompt)
def my_prompt_fn(r: pd.Series) -> ChatRequest:
    message = Message(
        role="user",
        content=f"Hello ChatGPT, Give a brief overview of the book {r.book_name}.",
    )

[//]: # (key Dict is mandatory since results order is NOT guaranteed!)
    key = {"user_id": r.user_id, "book_id": r.book_id}
    return ChatRequest(
        key=key,
        messages=[message],
        system=SystemMessage(content="Assistant is providing book reviews"),
    )

[//]: # (parallel process the DataFrame making up to 4 concurrent requests to OpenAI endpoint)
result_df = client.chat_completions(df=input_df, request_fn=my_prompt_fn,max_connections=4)

[//]: # (result_df columns contains 'openai_reply' and 'api_error' columns.

````

### Default Response Extraction

By default, only the "assistant" message (or messages if n>1) would be returned,
but you can implement a custom ResponseProcessor

```
class ResponseProcessor(Generic[R], Callable[..., R], ABC):
    @abstractmethod
    def __call__(self, json: str, *args: Any, **kwargs: Any) -> R:
        pass
```

### Disclaimer
This repository has no connection whatsoever to OpenAI.
