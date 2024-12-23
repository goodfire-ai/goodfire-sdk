from unittest.mock import patch

from sdk.python.goodfire.api.chat.client import ChatAPI


def test_chat_completions_streaming():
    api_key = "test"
    base_url = "http://localhost:8000"

    chat_api = ChatAPI(api_key, base_url)

    messages = [
        {
            "role": "user",
            "content": "Hello, how are you?",
        },
        {
            "role": "assistant",
            "content": "I'm good, how about you?",
        },
    ]
    model = "goodfire/agi"

    with patch("httpx.Client.stream") as mock_stream:
        mock_stream.return_value.__enter__.return_value.iter_bytes = lambda: iter(
            [
                b'data: {"object": "chat_completion", "id": "1", "created": 1630000000, "model": "goodfire/agi", "system_fingerprint": "", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "I\'m"}, "gf_token_index": 0, "finish_reason": "stop"}]}\n',
                b'data: {"object": "chat_completion", "id": "2", "created": 1630000001, "model": "goodfire/agi", "system_fingerprint": "", "choices": [{"index": 0, "delta": {"role": "assistant", "content": " good"}, "gf_token_index": 1, "finish_reason": "stop"}]}\n',
                b"data: [DONE]\n\n",
            ]
        )

        mock_stream.return_value.__enter__.return_value.status_code = 200

        for token in chat_api.completions.create(messages, model, stream=True):
            assert token.object == "chat_completion"
            assert token.model == "goodfire/agi"
            assert len(token.choices) == 1
            assert token.choices[0].delta.role == "assistant"
            assert token.choices[0].delta.content is not None
            assert token.choices[0].finish_reason == "stop"


# def test_chat_completions():
#     api_key = "test"
#     base_url = "http://localhost:8000"

#     chat_api = ChatAPI(api_key, base_url)

#     messages = [
#         {
#             "role": "user",
#             "content": "Hello, how are you?",
#         },
#         {
#             "role": "assistant",
#             "content": "I'm good, how about you?",
#         },
#     ]
#     model = "goodfire/agi"

#     with patch("sdk.python.goodfire.api.chat.client.run_async_safely") as mock_post:
#         mock_post.return_value = {
#             "object": "chat_completion",
#             "id": "1",
#             "created": 1630000000,
#             "model": "goodfire/agi",
#             "system_fingerprint": "",
#             "choices": [
#                 {
#                     "index": 0,
#                     "message": {
#                         "role": "assistant",
#                         "content": "I'm good, how about you?",
#                     },
#                     "finish_reason": "stop",
#                 }
#             ],
#         }

#         completion = chat_api.completions.create(messages, model)

#         assert completion.object == "chat_completion"
#         assert completion.model == "goodfire/agi"
#         assert len(completion.choices) == 1
#         assert completion.choices[0].message["role"] == "assistant"
#         assert completion.choices[0].message["content"] == "I'm good, how about you?"
#         assert completion.choices[0].finish_reason == "stop"
