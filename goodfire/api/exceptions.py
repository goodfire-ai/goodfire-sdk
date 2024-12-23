class GoodfireBaseException(Exception):
    pass


class RateLimitException(GoodfireBaseException):
    pass


class InvalidRequestException(GoodfireBaseException):
    pass


class ForbiddenException(GoodfireBaseException):
    pass


class NotFoundException(GoodfireBaseException):
    pass


class UnauthorizedException(GoodfireBaseException):
    pass


class ServerErrorException(GoodfireBaseException):
    pass


class RequestFailedException(GoodfireBaseException):
    pass


class InsufficientFundsException(GoodfireBaseException):
    pass


def check_status_code(status_code: int, respone_text: str):
    if status_code == 400:
        raise InvalidRequestException(respone_text or "Bad request.").with_traceback(
            None
        )
    elif status_code == 401:
        raise UnauthorizedException(respone_text or "Invalid API key.").with_traceback(
            None
        )
    elif status_code == 402:
        raise InsufficientFundsException(
            respone_text or "Insufficient credits."
        ).with_traceback(None)
    elif status_code == 403:
        raise ForbiddenException(
            respone_text or "Insufficient permissions."
        ).with_traceback(None)
    elif status_code == 404:
        raise NotFoundException(respone_text or "Not found.").with_traceback(None)
    elif status_code == 429:
        raise RateLimitException(
            respone_text
            or "You have hit your rate limit. You can request a higher limit by contacting the Goodfire team."
        ).with_traceback(None)
    elif status_code == 500:
        raise ServerErrorException("Server error").with_traceback(None)
    elif status_code > 400:
        raise RequestFailedException(respone_text or "Bad request").with_traceback(None)
