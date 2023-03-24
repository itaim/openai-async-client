import time

from openai_async_client.reader import Usage,Completion

# prompt_tokens: int
#     completion_tokens: int
#     total_tokens: int

comp = Completion(id="com-999",created=time.time(),text="hello word",usage=Usage(prompt_tokens=100,completion_tokens=67,total_tokens=167))

print(comp.dict())
d = comp.dict()
d.update({"a":1,"b":3})
print(d)