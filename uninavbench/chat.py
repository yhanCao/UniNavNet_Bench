"""Chat completion requests that work across OpenAI-compatible providers."""
import openai


def create_chat(client, **kwargs):
    """client.chat.completions.create, adapting to what the provider accepts.

    Newer OpenAI models reject ``max_tokens`` (they take ``max_completion_tokens``) and any temperature other
    than the default, while other OpenAI-compatible servers may not know ``max_completion_tokens``. A request
    rejected for one of these parameters is retried with it renamed or dropped.
    """
    for _ in range(3):
        try:
            return client.chat.completions.create(**kwargs)
        except openai.BadRequestError as ex:
            message = str(ex)
            if "max_tokens" in kwargs and "max_tokens" in message:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            elif "temperature" in kwargs and "temperature" in message:
                kwargs.pop("temperature")
            else:
                raise
    return client.chat.completions.create(**kwargs)
