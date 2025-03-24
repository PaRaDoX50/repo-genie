import sys

import numpy as np
from colorama import Fore, Style

from assistant.index import search_index


class BaseAssistant:
    """
    A base class for LLM assistants that enables inclusion of local files in the LLM context. Files
    are collected recursively from the current directory.
    """

    def __init__(
        self,
        system_instructions,
        embed,
        index,
        chunks,
        context_file_ratio,
        output_acceptance_retries
    ):
        self.system_instructions = system_instructions
        self.embed = embed
        self.index = index
        self.chunks = chunks
        self.context_file_ratio = context_file_ratio
        self.context_size = 8192
        self.output_acceptance_retries = output_acceptance_retries

    def initialize_history(self):
        system_instructions_tokens = self.count_tokens(self.system_instructions)
        self.chat_history = [
            {
                "role": "system",
                "content": self.system_instructions,
                "tokens": system_instructions_tokens,
            }
        ]

    def call_completion(self, chat_history):
        # unimplemented on base class
        raise NotImplementedError

    def count_tokens(self, text):
        # unimplemented on base class
        raise NotImplementedError

    def build_relevant_full_text(self, user_input):
        relevant_chunks = search_index(self.embed, self.index, user_input, self.chunks)
        relevant_full_text = ""
        chunk_total_tokens = 0
        for i, relevant_chunk in enumerate(relevant_chunks, start=1):
            # Note: relevant_chunk["tokens"] is created with the embedding model, not the LLM, so it will
            # not be accurate for the purposes of maximizing the context of the LLM.
            chunk_total_tokens += self.count_tokens(relevant_chunk["text"] + "\n\n")
            if chunk_total_tokens >= self.context_size * self.context_file_ratio:
                break
            relevant_full_text += relevant_chunk["text"] + "\n\n"
        return relevant_full_text

    def create_user_history(self, temp_content, final_content):
        return {
            "role": "user",
            "content": temp_content,
            "tokens": self.embed.count_tokens(final_content),
        }

    def add_user_history(self, temp_content, final_content):
        self.chat_history.append(self.create_user_history(temp_content, final_content))

    def cull_history(self):
        self.cull_history_list(self.chat_history)

    def cull_history_list(self, history_list):
        sum_of_tokens = sum(
            [self.count_tokens(message["content"]) for message in history_list]
        )
        while sum_of_tokens > self.context_size:
            history_list.pop(0)
            sum_of_tokens = sum(
                [self.count_tokens(message["content"]) for message in history_list]
            )

        # Some LLMs require the first message to be from the user
        if history_list[0]["role"] == "system":
            while len(history_list) > 1 and history_list[1]["role"] == "assistant":
                history_list.pop(1)
        else:
            while len(history_list) > 0 and history_list[0]["role"] == "assistant":
                history_list.pop(0)

    def create_empty_history(self, role="user"):
        return {"role": role, "content": "", "tokens": 0}

    def create_prompt(self, user_input):
        return user_input

    def run_stream_processes(self, user_input):
        # Returns a string of the assistant's response
        prompt = self.create_prompt(user_input)
        relevant_full_text = self.build_relevant_full_text(prompt)
        return self.run_basic_chat_stream(prompt, relevant_full_text)

    def run_post_stream_processes(self, user_input, stream_output):
        # Returns whether the output should be accepted
        return True

    def run_basic_chat_stream(self, user_input, relevant_full_text):
        # Add the user input to the chat history
        user_content = relevant_full_text + user_input
        self.add_user_history(user_content, user_input)

        self.cull_history()
        completion_generator = self.call_completion(self.chat_history)

        # Replace the RAG output with the user input. This reduces the size of the history for future prompts.
        self.chat_history[-1]["content"] = user_input

        # Display chat history
        output_history = self.create_empty_history()

        output_history = self.run_completion_generator(
            completion_generator, output_history
        )

        # Add the completion to the chat history
        output_history["tokens"] = self.count_tokens(output_history["content"])
        self.chat_history.append(output_history)
        return output_history["content"]

    def create_thinking_context(self):
        return {
            "thinking_start_finished": not self.hide_thinking,
            "thinking_end_finished": not self.hide_thinking,
            "delta_after_thinking_finished": None,
        }

    def run_completion_generator(
        self, completion_output, output_message
    ):
        for chunk in completion_output:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"] != None:
                output_message["content"] += delta["content"]
        return output_message