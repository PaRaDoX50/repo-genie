from typing import Any
from mcp.server.fastmcp import FastMCP
import os
from typing import Dict
import sys
from cli.config import load_config
from cli.start import initialize_llm
from assistant.cgrag_assistant import CGRAGAssistant # Import CGRAGAssistant for type checking


# Initialize FastMCP server
mcp = FastMCP("weather", port = 9191)


# async def make_nws_request(url: str) -> dict[str, Any] | None:
#     """Make a request to the NWS API with proper error handling."""
#     headers = {
#         "User-Agent": USER_AGENT,
#         "Accept": "application/geo+json"
#     }
#     async with httpx.AsyncClient() as client:
#         try:
#             response = await client.get(url, headers=headers, timeout=30.0)
#             response.raise_for_status()
#             return response.json()
#         except Exception:
#             return None

# def format_alert(feature: dict) -> str:
#     print(feature)
#     """Format an alert feature into a readable string."""
#     props = feature["properties"]
#     return f"""
# Event: {props.get('event', 'Unknown')}
# Area: {props.get('areaDesc', 'Unknown')}
# Severity: {props.get('severity', 'Unknown')}
# Description: {props.get('description', 'No description available')}
# Instructions: {props.get('instruction', 'No specific instructions provided')}
# """


# @mcp.tool()
# async def get_alerts(state: str) -> str:
#     """Get weather alerts for a US state.

#     Args:
#         state: Two-letter US state code (e.g. CA, NY)
#     """

#     print("sdfasf")
#     url = f"{NWS_API_BASE}/alerts/active/area/{state}"
#     data = await make_nws_request(url)

#     if not data or "features" not in data:
#         return "Unable to fetch alerts or no alerts found."

#     if not data["features"]:
#         return "No active alerts for this state."

#     alerts = [format_alert(feature) for feature in data["features"]]
#     return "\n---\n".join(alerts)

# @mcp.tool()
# async def get_forecast(latitude: float, longitude: float) -> str:
#     """Get weather forecast for a location.

#     Args:
#         latitude: Latitude of the location
#         longitude: Longitude of the location
#     """
#     print("sdsdfsdfsdfasf")
#     # First get the forecast grid endpoint
#     points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
#     points_data = await make_nws_request(points_url)

#     if not points_data:
#         return "Unable to fetch forecast data for this location."

#     # Get the forecast URL from the points response
#     forecast_url = points_data["properties"]["forecast"]
#     forecast_data = await make_nws_request(forecast_url)

#     if not forecast_data:
#         return "Unable to fetch detailed forecast."

#     # Format the periods into a readable forecast
#     periods = forecast_data["properties"]["periods"]
#     forecasts = []
#     for period in periods[:5]:  # Only show next 5 periods
#         forecast = f"""
# {period['name']}:
# Temperature: {period['temperature']}°{period['temperatureUnit']}
# Wind: {period['windSpeed']} {period['windDirection']}
# Forecast: {period['detailedForecast']}
# """
#         forecasts.append(forecast)

#     return "\n---\n".join(forecasts)

llm = None  # Global variable to store the initialized LLM assistant
hardcoded_directories = []

def startup_event():
    """
    Initializes the LLM assistant and indexes the hardcoded directories when the server starts.
    """
    global llm, hardcoded_directories

    # Load the configuration
    config_dict = load_config()

    hardcoded_directories = ["/Users/suryanshtomar/Desktop/scapia-projs/backend/lollllll"]

    # Initialize the LLM assistant
    try:
        # Create dummy args for command line
        class Args:
            ignore = []
            dirs = hardcoded_directories
            single_prompt = None
            verbose = False
            no_color = True  # Set no_color to True for server use
        args = Args()
        print("Initializing assistant...")
        llm = initialize_llm(args, config_dict)
        print("Assistant initialized successfully.")

        # print({}f"Indexing directories: {hardcoded_directories}")

    except Exception as e:
        print(f"Error during startup: {e}", file=sys.stderr)
        raise  # Re-raise the exception to prevent the server from starting

@mcp.tool()
async def ask_question(user_prompt:str) -> Dict[str, str]:
    """
    Accepts a user prompt to understand code and returns the LLM's answer.
    """
    global llm
    if llm is None:
        return "The assistant is not initialized yet. Please try again later."

    try:
        llm.initialize_history()
        response = llm.run_stream_processes(user_prompt)

        # Only print the final response
        # sys.stdout.write(response)
        # answer = llm.run_stream_processes(user_prompt)
        return response
    
    except Exception as e:
        return f"Error during question answering: {e}"


if __name__ == "__main__":
    try:
        startup_event()
        mcp.run(transport='stdio')
    except Exception as e:
        print(f"Error during server startup: {e}", file=sys.stderr)