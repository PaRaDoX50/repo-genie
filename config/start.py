import litellm

from assistant.index import create_file_index
from assistant.lite_llm_assistant import LiteLLMAssistant
from assistant.lite_llm_embed import LiteLlmEmbed
from config.config import (
    get_file_path,
)

litellm.suppress_debug_info = True

def initialize_llm(args, config_dict):
    config = (
        config_dict["CONF"] if "CONF" in config_dict else config_dict
    )

    context_file_ratio = config["CONTEXT_FILE_RATIO"]
    system_instructions = f"""{config["SYSTEM_INSTRUCTIONS"]}"""
    
    # Llama.cpp settings
    # llm_model_file = get_file_path(config["MODELS_PATH"], config["LLM_MODEL"])
    # embed_model_file = get_file_path(config["MODELS_PATH"], config["EMBED_MODEL"])

    # LiteLLM settings
    lite_llm_context_size = config["LITELLM_CONTEXT_SIZE"]
    lite_llm_embed_context_size = config["LITELLM_EMBED_CONTEXT_SIZE"]
    lite_llm_completion_options = config["LITELLM_COMPLETION_OPTIONS"]
    lite_llm_embed_completion_options = config["LITELLM_EMBED_COMPLETION_OPTIONS"]
    lite_llm_model_uses_system_message = config["LITELLM_MODEL_USES_SYSTEM_MESSAGE"]
    lite_llm_pass_through_context_size = config["LITELLM_PASS_THROUGH_CONTEXT_SIZE"]
    lite_llm_embed_request_delay = float(config["LITELLM_EMBED_REQUEST_DELAY"])

    # Assistant settings
    use_cgrag = args.use_cgrag
    output_acceptance_retries = config["OUTPUT_ACCEPTANCE_RETRIES"]

    if "model" not in lite_llm_completion_options or not lite_llm_completion_options["model"]:
        return """You must specify LITELLM_COMPLETION_OPTIONS.model."""
    if "model" not in lite_llm_embed_completion_options or not lite_llm_embed_completion_options["model"]:
        return """You must specify LITELLM_EMBED_COMPLETION_OPTIONS.model."""

    ignore_paths = args.ignore if args.ignore else []
    ignore_paths.extend(config["GLOBAL_IGNORES"])

    extra_dirs = args.dirs if args.dirs else []

    embed = LiteLlmEmbed(
        lite_llm_embed_completion_options=lite_llm_embed_completion_options,
        lite_llm_embed_context_size=lite_llm_embed_context_size,
        delay=lite_llm_embed_request_delay,
    )
    embed_chunk_size = lite_llm_embed_context_size

    index, chunks = create_file_index(
        embed, ignore_paths, embed_chunk_size, extra_dirs
    )

    llm = LiteLLMAssistant(
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
    )

    return llm