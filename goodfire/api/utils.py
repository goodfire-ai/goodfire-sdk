import time
from typing import Any, AsyncIterator, Iterator, Optional

import httpx

from ..utils.asyncio import run_async_safely
from ..utils.logger import logger
from .exceptions import RateLimitException, check_status_code


class HTTPWrapper:
    def __init__(self, inital_backoff_time: float = 1.3, max_retries: int = 5):
        self._async_http = AsyncHTTPWrapper(inital_backoff_time, max_retries)

    def get(
        self,
        url: str,
        headers: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = 10,
    ) -> httpx.Response:
        return run_async_safely(
            self._async_http.get(url, headers, params, timeout=timeout)
        )

    def post(
        self,
        url: str,
        headers: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = 10,
    ) -> httpx.Response:
        return run_async_safely(
            self._async_http.post(url, headers, json, timeout=timeout)
        )

    def put(
        self,
        url: str,
        headers: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = 10,
    ) -> httpx.Response:
        return run_async_safely(
            self._async_http.put(url, headers, json, timeout=timeout)
        )

    def delete(
        self,
        url: str,
        headers: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = 10,
    ) -> httpx.Response:
        return run_async_safely(self._async_http.delete(url, headers, timeout=timeout))

    def stream(
        self,
        method: str,
        url: str,
        headers: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = 10,
        _attempt_num: int = 0,
    ) -> Iterator[bytes]:
        def _stream_response():
            with httpx.Client() as client:
                try:
                    with client.stream(
                        method,
                        url,
                        headers=headers,
                        json=json,
                        params=params,
                        timeout=timeout,
                    ) as response:
                        if response.status_code != 200:
                            response.read()
                            check_status_code(response.status_code, response.text)
                        else:
                            for bytes in response.iter_bytes():
                                yield bytes
                except RateLimitException:
                    if _attempt_num < self._async_http.max_retries:
                        self._async_http._rate_limit_warning()
                        time.sleep(
                            self._async_http.inital_backoff_time**_attempt_num + 1
                        )
                        result = self.stream(
                            method,
                            url,
                            headers,
                            json,
                            params,
                            timeout=timeout,
                            _attempt_num=_attempt_num + 1,
                        )

                        for bytes in result:
                            yield bytes
                    else:
                        raise RateLimitException("Rate limit exceeded")

        return _stream_response()


class AsyncHTTPWrapper:
    def __init__(self, inital_backoff_time: float = 2, max_retries: int = 5):
        self.inital_backoff_time = inital_backoff_time
        self.max_retries = max_retries

    def _rate_limit_warning(self):
        logger.warning("Rate limit exceeded. Attempting exponential backoff...")

    async def get(
        self,
        url: str,
        headers: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        _attempt_num: int = 0,
        timeout: Optional[int] = 10,
    ) -> httpx.Response:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url, headers=headers, params=params, timeout=timeout
            )
            try:
                check_status_code(response.status_code, response.text)
            except RateLimitException:
                if _attempt_num < self.max_retries:
                    self._rate_limit_warning()
                    time.sleep(self.inital_backoff_time**_attempt_num)
                    return await self.get(
                        url, headers, params, _attempt_num + 1, timeout=timeout
                    )
                else:
                    raise RateLimitException("Rate limit exceeded")

            return response

    async def post(
        self,
        url: str,
        headers: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
        _attempt_num: int = 0,
        timeout: Optional[int] = 10,
    ) -> httpx.Response:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, headers=headers, json=json, timeout=timeout
            )
            try:
                check_status_code(response.status_code, response.text)
            except RateLimitException:
                if _attempt_num < self.max_retries:
                    self._rate_limit_warning()
                    time.sleep(self.inital_backoff_time ** (_attempt_num + 1))
                    return await self.post(
                        url, headers, json, _attempt_num + 1, timeout=timeout
                    )
                else:
                    raise RateLimitException("Rate limit exceeded")

            return response

    async def put(
        self,
        url: str,
        headers: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
        _attempt_num: int = 0,
        timeout: Optional[int] = 10,
    ) -> httpx.Response:
        async with httpx.AsyncClient() as client:
            response = await client.put(
                url, headers=headers, json=json, timeout=timeout
            )
            try:
                check_status_code(response.status_code, response.text)
            except RateLimitException:
                if _attempt_num < self.max_retries:
                    self._rate_limit_warning()
                    time.sleep(self.inital_backoff_time**_attempt_num)
                    return await self.put(
                        url, headers, json, _attempt_num + 1, timeout=timeout
                    )
                else:
                    raise RateLimitException("Rate limit exceeded")

            return response

    async def delete(
        self,
        url: str,
        headers: Optional[dict[str, Any]] = None,
        _attempt_num: int = 0,
        timeout: Optional[int] = 10,
    ) -> httpx.Response:
        async with httpx.AsyncClient() as client:
            response = await client.delete(url, headers=headers, timeout=timeout)
            try:
                check_status_code(response.status_code, response.text)
            except RateLimitException:
                if _attempt_num < self.max_retries:
                    self._rate_limit_warning()
                    time.sleep(self.inital_backoff_time**_attempt_num)
                    return await self.delete(
                        url, headers, _attempt_num + 1, timeout=timeout
                    )
                else:
                    raise RateLimitException("Rate limit exceeded")

            return response

    async def stream(
        self,
        method: str,
        url: str,
        headers: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = 10,
        _attempt_num: int = 0,
    ) -> AsyncIterator[bytes]:
        async def _stream_response():
            async with httpx.AsyncClient() as client:
                try:
                    async with client.stream(
                        method,
                        url,
                        headers=headers,
                        json=json,
                        params=params,
                        timeout=timeout,
                    ) as response:
                        if response.status_code != 200:
                            response.read()
                            check_status_code(response.status_code, response.text)
                        else:
                            async for bytes in response.aiter_bytes():
                                yield bytes
                except RateLimitException:
                    if _attempt_num < self.max_retries:
                        self._rate_limit_warning()
                        time.sleep(self.inital_backoff_time**_attempt_num + 1)
                        result: AsyncIterator[bytes] = _stream_response()

                        async for bytes in result:
                            yield bytes
                    else:
                        raise RateLimitException("Rate limit exceeded")

        return _stream_response()
