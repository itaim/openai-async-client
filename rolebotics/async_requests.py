import asyncio
import logging
from typing import List, Dict, Any, Optional, Union

import httpx
from backoff import on_exception, expo
from httpx import Timeout


class PostRequest(object):
    def __init__(self, key: Dict[str, Any], body=Dict[str, Any]):
        self.key = key
        self.body = body


class PostResult(object):
    def __init__(
        self,
        key: Dict[str, Any],
        value=Optional[str],
        error: Optional[BaseException] = None,
    ):
        self.key = key
        self.value = value
        self.error = error


class HttpxErrorResponse(Exception):
    def __init__(self, response: httpx.Response):
        super().__init__(response.text)
        self.status_code = response.status_code
        self.response = response


async def post_request(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    request: PostRequest,
    max_retries: int = 5,
) -> PostResult:
    @on_exception(expo, httpx.RequestError, max_tries=max_retries)
    async def make_request(request: PostRequest) -> PostResult:
        response = await client.post(url, json=request.body, headers=headers)
        if response.status_code != 200:
            raise httpx.RequestError("Failed request", request=response.request)
        logging.debug(response.text)
        return PostResult(key=request.key, value=response.text, error=None)

    try:
        return await make_request(request)
    except httpx.RequestError as e:
        logging.exception(e)
        return PostResult(key=request.key, value=None, error=e)


async def process_payloads(
    endpoint: str,
    headers: Dict[str, str],
    requests: List[PostRequest],
    max_concurrent_connections: int = 5,
    max_retries: int = 5,
    timeout: Optional[Timeout] = None,
) -> List[Union[PostResult, BaseException]]:
    async with httpx.AsyncClient(
        timeout=timeout, limits=httpx.Limits(max_connections=max_concurrent_connections)
    ) as client:
        tasks = [
            post_request(client, endpoint, headers, request, max_retries)
            for request in requests
        ]
        return list(await asyncio.gather(*tasks))
