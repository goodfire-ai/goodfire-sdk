import pytest

from sdk.python.goodfire.api.exceptions import (
    ForbiddenException,
    InsufficientFundsException,
    InvalidRequestException,
    NotFoundException,
    RateLimitException,
    RequestFailedException,
    ServerErrorException,
    UnauthorizedException,
    check_status_code,
)


def test_exceptions():
    with pytest.raises(ForbiddenException):
        check_status_code(403, "hello")

    with pytest.raises(NotFoundException):
        check_status_code(404, "hello")

    with pytest.raises(RateLimitException):
        check_status_code(429, "hello")

    with pytest.raises(InvalidRequestException):
        check_status_code(400, "hello")

    with pytest.raises(ServerErrorException):
        check_status_code(500, "hello")

    with pytest.raises(UnauthorizedException):
        check_status_code(401, "hello")

    with pytest.raises(InsufficientFundsException):
        check_status_code(402, "hello")

    with pytest.raises(RequestFailedException):
        check_status_code(482, "hello")
