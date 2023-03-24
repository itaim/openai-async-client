# OpenAI client with client timeout and parallel processing

## Quick Install

`pip install openai-async-client`

## 🤔 What is this?

This library is aimed at assisting with OpenAI API usage by:

Support for client side timeouts with retry and backoff for completions.

Support for concurrent processing with pandas DataFrames.

##

### Chat Completion
Example of chat completion with client timeout of 1 second to connect and 10 seconds to read with a maximum of 3
retries.

````
import os
from httpx import Timeout
from openai_async_client import AsyncCreate, Message, ChatCompletionRequest, SystemMessage, OpenAIParams
from openai_async_client.model import TextCompletionRequest, EndpointConfig

from openai_async_client import OpenAIAsync, ChatRequest, Message, SystemMessage

create = AsyncCreate(api_key=os.environ["OPENAI_API_KEY"])
messages = [
    Message(
        role="user",
        content=f"Hello ChatGPT, Give a brief overview of the Pride and Prejudice by Jane Austen.",
    )
]
response = create.completion(ChatCompletionRequest(prompt=messages),client_timeout=Timeout(1.0,read=10.0),retries=3)

````


### Text Completion.

````
create = AsyncCreate()
response = create.completion(TextCompletionRequest(prompt=f"Hello DaVinci, Give a brief overview of Moby Dick by  Herman Melville."))
````

### DataFrame processing
Example of parallel chat completions for a DataFrame  with concurrent connections.

 ````
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


create = AsyncCreate()

[//]: # (Define a mapping function from row to prompt)
def chat_prompt_fn(r: pd.Series) -> ChatCompletionRequest:
    message = Message(
        role="user",
        content=f"Hello ChatGPT, Give a brief overview of the book {r.book_name}.",
    )
    [//]: # (key Dict is mandatory since results are NOT returned in order)
    key = {"user_id": r.user_id, "book_id": r.book_id}
    return ChatCompletionRequest(
        key=key,
        prompt=[message],
        system=SystemMessage(content="Assistant is providing book reviews"),
        params=OpenAIParams(model="gpt-3.5-turbo", n=n)
    )

[//]: # (parallel process the DataFrame making up to max_connections concurrent requests to OpenAI endpoint)
result_df = client.chat_completions(df=input_df, request_fn=chat_prompt_fn,max_connections=4)

[//]: # (result_df columns are the input_df columns plus:
 "openai_completion" - the completion/s (str/list),
 "openai_id",
 "openai_created",
 "openai_prompt_tokens",
 "openai_completion_tokens",
 "openai_total_tokens",
 "api_error" - http, openai, api error or pd.NA when everything is Okay.

````


### Disclaimer
This repository has no connection to OpenAI.
