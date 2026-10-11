from langfuse import observe


class SystemPrompt:
    X_PROMPT_TEMPLATE: str | None = None

    @classmethod
    @observe
    def get_x_prompt_template(cls) -> str:
        if cls.X_PROMPT_TEMPLATE is None:
            cls.X_PROMPT_TEMPLATE = """write prompt {placeholder} here"""

        return cls.X_PROMPT_TEMPLATE
