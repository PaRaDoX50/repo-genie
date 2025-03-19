from typing import Any
from mcp.server.fastmcp import FastMCP
import os
from typing import Dict
import sys
from config.config import load_config
from config.start import initialize_llm
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn
from starlette.middleware.cors import CORSMiddleware


# Initialize FastMCP server
mcp = FastMCP("weather")

card_llm = None
reward_llm = None
bus_llm = None
orders_llm = None
hotels_llm = None
faq_llm = None

def startup_event_reward():
    """
    Initializes the LLM assistant and indexes the rewards service when the server starts.
    """
    global reward_llm

    # Load the configuration
    config_dict = load_config()

    hardcoded_directories = ["/Users/sumedhzope/dir_test/rewards-service"]

    # Initialize the LLM assistant
    try:
        # Create dummy args for command line
        class Args:
            ignore = []
            dirs = hardcoded_directories
            single_prompt = None
            verbose = False
            no_color = True  # Set no_color to True for server use
            use_cgrag = True
        args = Args()
        print("Initializing rewards assistant...")
        reward_llm = initialize_llm(args, config_dict)
        print("Rewards Assistant initialized successfully.")

    except Exception as e:
        print(f"Error during startup: {e}", file=sys.stderr)
        raise  # Re-raise the exception to prevent the server from starting

def startup_event_cards():
    """
    Initializes the LLM assistant and indexes the cards service when the server starts.
    """
    global card_llm

    # Load the configuration
    config_dict = load_config()

    hardcoded_directories = ["/Users/sumedhzope/dir_test/cards-service"]

    # Initialize the LLM assistant
    try:
        # Create dummy args for command line
        class Args:
            ignore = []
            dirs = hardcoded_directories
            single_prompt = None
            verbose = False
            no_color = True  # Set no_color to True for server use
            use_cgrag = True
        args = Args()
        print("Initializing cards assistant...")
        card_llm = initialize_llm(args, config_dict)
        print("Cards Assistant initialized successfully.")

    except Exception as e:
        print(f"Error during startup: {e}", file=sys.stderr)
        raise  # Re-raise the exception to prevent the server from starting

def startup_event_bus():
    """
    Initializes the LLM assistant and indexes the bus service when the server starts.
    """
    global bus_llm

    # Load the configuration
    config_dict = load_config()

    hardcoded_directories = ["/Users/sumedhzope/dir_test/bus-service"]

    # Initialize the LLM assistant
    try:
        # Create dummy args for command line
        class Args:
            ignore = []
            dirs = hardcoded_directories
            single_prompt = None
            verbose = False
            no_color = True  # Set no_color to True for server use
            use_cgrag = True
        args = Args()
        print("Initializing bus assistant...")
        bus_llm = initialize_llm(args, config_dict)
        print("Bus Assistant initialized successfully.")

    except Exception as e:
        print(f"Error during startup: {e}", file=sys.stderr)
        raise  # Re-raise the exception to prevent the server from starting

def startup_event_orders():
    """
    Initializes the LLM assistant and indexes the orders service when the server starts.
    """
    global orders_llm

    # Load the configuration
    config_dict = load_config()

    hardcoded_directories = ["/Users/sumedhzope/dir_test/orders"]

    # Initialize the LLM assistant
    try:
        # Create dummy args for command line
        class Args:
            ignore = []
            dirs = hardcoded_directories
            single_prompt = None
            verbose = False
            no_color = True  # Set no_color to True for server use
            use_cgrag = True
        args = Args()
        print("Initializing orders assistant...")
        orders_llm = initialize_llm(args, config_dict)
        print("Orders Assistant initialized successfully.")

    except Exception as e:
        print(f"Error during startup: {e}", file=sys.stderr)
        raise  # Re-raise the exception to prevent the server from starting

def startup_event_hotels():
    """
    Initializes the LLM assistant and indexes the hotels service when the server starts.
    """
    global hotels_llm

    # Load the configuration
    config_dict = load_config()

    hardcoded_directories = ["/Users/sumedhzope/dir_test/hotel-service"]

    # Initialize the LLM assistant
    try:
        # Create dummy args for command line
        class Args:
            ignore = []
            dirs = hardcoded_directories
            single_prompt = None
            verbose = False
            no_color = True  # Set no_color to True for server use
            use_cgrag = True
        args = Args()
        print("Initializing hotels assistant...")
        hotels_llm = initialize_llm(args, config_dict)
        print("Hotels Assistant initialized successfully.")

    except Exception as e:
        print(f"Error during startup: {e}", file=sys.stderr)
        raise  # Re-raise the exception to prevent the server from starting

def startup_event_faqs():
    """
    Initializes the LLM assistant and indexes the faq service when the server starts.
    """
    global faq_llm

    # Load the configuration
    config_dict = load_config()

    hardcoded_directories = ["/Users/sumedhzope/dir_test/faqs_test"]

    # Initialize the LLM assistant
    try:
        # Create dummy args for command line
        class Args:
            ignore = []
            dirs = hardcoded_directories
            single_prompt = None
            verbose = False
            no_color = True  # Set no_color to True for server use
            use_cgrag = False
        args = Args()
        print("Initializing faqs assistant...")
        faq_llm = initialize_llm(args, config_dict)
        print("Faqs Assistant initialized successfully.")

    except Exception as e:
        print(f"Error during startup: {e}", file=sys.stderr)
        raise  # Re-raise the exception to prevent the server from starting

@mcp.tool()
async def ask_rewards_service(user_prompt:str) -> Dict[str, str]:
    """
    Rewards Service Tool
    A comprehensive rewards management system that handles Scapia Coins, tiered loyalty programs (Silver/Gold/Platinum), 
    personalized offers, TNPL integration, badge rewards, transaction-based crediting, statements, lounge access, referrals, 
    joining bonuses, merchant-specific rewards, and transaction history visualization.

    Args:
        user_prompt: The user's question or prompt.
    """
    global reward_llm
    if reward_llm is None:
        return "The assistant is not initialized yet. Please try again later."

    try:
        reward_llm.initialize_history()
        response = reward_llm.run_stream_processes(user_prompt)
        return response
    
    except Exception as e:
        return f"Error during question answering: {e}"
    
@mcp.tool()
async def ask_cards_service(user_prompt:str) -> Dict[str, str]:
    """
    Cards Service Tool
    Complete card lifecycle management including digital/physical issuance, PIN controls, contactless settings, card status management (lock/unlock/hotlist), 
    replacement workflows, add-on cards, limit customization, transaction monitoring, tokenization, secure CVV retrieval, and dedicated CX support tools.

    Args:
        user_prompt: The user's question or prompt.
    """
    global card_llm
    if card_llm is None:
        return "The assistant is not initialized yet. Please try again later."

    try:
        card_llm.initialize_history()
        response = card_llm.run_stream_processes(user_prompt)
        return response
    
    except Exception as e:
        return f"Error during question answering: {e}"

@mcp.tool()
async def ask_bus_service(user_prompt:str) -> Dict[str, str]:
    """
    Bus Service Tool
    End-to-end bus booking platform with multi-provider search, seat visualization, comprehensive filtering (routes/operators/amenities), 
    EMI options, coin payments, automated cancellations/refunds, journey tracking, digital tickets, pickup mapping, 
    and real-time journey assistance.

    Args:
        user_prompt: The user's question or prompt.
    """
    global bus_llm
    if bus_llm is None:
        return "The assistant is not initialized yet. Please try again later."

    try:
        bus_llm.initialize_history()
        response = bus_llm.run_stream_processes(user_prompt)
        return response
    
    except Exception as e:
        return f"Error during question answering: {e}"
    
@mcp.tool()
async def ask_orders_service(user_prompt:str) -> Dict[str, str]:
    """
    Orders Service Tool
    Unified order management system supporting cross-product tracking, multiple payment methods, TNPL/EMI plans, coin payments, 
    dynamic discounts, prorated refunds, automated cancellations, reference tracking, trip bundling, TCS calculation, payment links, 
    and comprehensive order history.

    Args:
        user_prompt: The user's question or prompt.
    """
    global orders_llm
    if orders_llm is None:
        return "The assistant is not initialized yet. Please try again later."

    try:
        orders_llm.initialize_history()
        response = orders_llm.run_stream_processes(user_prompt)
        return response
    
    except Exception as e:
        return f"Error during question answering: {e}"
    
@mcp.tool()
async def ask_hotels_service(user_prompt:str) -> Dict[str, str]:
    """
    Hotels Service Tool
    Hotel aggregation platform featuring multi-supplier integration, locality search, comprehensive filtering (amenities/stars/price), 
    personalized ranking, image galleries, verified reviews, room comparison, perks visualization, availability calendars, 
    price alerts, map view, and supplier-specific discounts.

    Args:
        user_prompt: The user's question or prompt.
    """
    global hotels_llm
    if hotels_llm is None:
        return "The assistant is not initialized yet. Please try again later."

    try:
        hotels_llm.initialize_history()
        response = hotels_llm.run_stream_processes(user_prompt)
        return response
    
    except Exception as e:
        return f"Error during question answering: {e}"
    
@mcp.tool()
async def ask_faqs(user_prompt:str) -> Dict[str, str]:
    """
    FAQs
    Use this mcp tool when any question is asked

    Args:
        user_prompt: The user's question or prompt.
    """
    global faq_llm
    if faq_llm is None:
        return "The assistant is not initialized yet. Please try again later."

    try:
        faq_llm.initialize_history()
        response = faq_llm.run_stream_processes(user_prompt)
        return response
    
    except Exception as e:
        return f"Error during question answering: {e}"

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that serves the provided MCP server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    app = Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3080"],  # Allow all origins (for development)
        allow_credentials=True,
        allow_methods=["*"],  # Allow all HTTP methods
        allow_headers=["*"],  # Allow all headers
    )

    return app


if __name__ == "__main__":
    startup_event_cards()
    startup_event_reward()
    startup_event_bus()
    startup_event_orders()
    startup_event_hotels()
    startup_event_faqs()
    mcp_server = mcp._mcp_server

    import argparse
    
    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    args = parser.parse_args()

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)