import asyncio
import threading
from queue import Queue
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_async_safely(coro: Coroutine[Any, Any, T]) -> T:
    """
    Safely runs a coroutine in a sync context, handling existing event loops.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Create a new event loop if one doesn't exist
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        result_queue: Queue[tuple[str, Any]] = Queue()

        async def run_in_new_loop():
            try:
                result = await coro
                result_queue.put(("result", result))
            except Exception as e:
                result_queue.put(("error", e))

        thread = threading.Thread(target=lambda: asyncio.run(run_in_new_loop()))
        thread.start()
        thread.join()
        status, value = result_queue.get()
        if status == "error":
            raise value

        return value

    return loop.run_until_complete(coro)
