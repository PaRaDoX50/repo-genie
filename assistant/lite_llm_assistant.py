import sys
from copy import deepcopy

from litellm import completion, token_counter

from assistant.cgrag_assistant import CGRAGAssistant


class LiteLLMAssistant(CGRAGAssistant):
    def __init__(
        self,
        lite_llm_completion_options,
        lite_llm_context_size,
        lite_llm_model_uses_system_message,
        lite_llm_pass_through_context_size,
        system_instructions,
        embed,
        index,
        chunks,
        context_file_ratio,
        output_acceptance_retries,
        use_cgrag
    ):
        super().__init__(
            system_instructions,
            embed,
            index,
            chunks,
            context_file_ratio,
            output_acceptance_retries,
            use_cgrag
        )
        self.completion_options = lite_llm_completion_options
        self.context_size = lite_llm_context_size
        self.pass_through_context_size = lite_llm_pass_through_context_size
        self.lite_llm_model_uses_system_message = lite_llm_model_uses_system_message

    def initialize_history(self):
        super().initialize_history()
        if not self.lite_llm_model_uses_system_message:
            self.chat_history[0]["role"] = "user"

    def call_completion(self, chat_history):
        # Clean "tokens" from chat history. It causes an error for mistral.
        chat_history_cleaned = deepcopy(chat_history)
        for message in chat_history_cleaned:
            if "tokens" in message:
                del message["tokens"]

        if self.pass_through_context_size:
            return completion(
                **self.completion_options,
                messages=chat_history_cleaned,
                stream=True,
                num_ctx=self.context_size,
            )
        else:
            return completion(
                **self.completion_options,
                messages=chat_history_cleaned,
                stream=True,
            )

    def count_tokens(self, text):
        return token_counter(
            model=self.completion_options["model"], messages=[{"user": "role", "content": text}]
        )
