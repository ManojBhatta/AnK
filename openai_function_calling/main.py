from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import requests
import json

# Load environment variables from .env file
load_dotenv()


# Define the function to get weather data
def get_current_weather(location):
    """Get the current weather in a given location"""
    base = "http://api.weatherapi.com/v1"
    key = os.getenv("WEATHER_API_KEY")
    request_url = f"{base}/current.json?key={key}&q={location}"
    response = requests.get(request_url)

    data = response.json()
    current = data["current"]
    location_info = data["location"]

    return {
        "location": location_info["name"],
        "country": location_info["country"],
        "temperature_c": current["temp_c"],
        "condition": current["condition"]["text"],
        "feels_like_c": current["feelslike_c"],
        "humidity": current["humidity"],
        "wind_kph": current["wind_kph"],
    }


# Create Azure OpenAI client
client = AzureOpenAI()

model_name = os.getenv("AZURE_DEPLOYMENT_NAME")


def run_conversation(content):
    messages = [{"role": "user", "content": content}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The name of place, can be with latitude  and longitudes.",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)

        available_functions = {
            "get_current_weather": get_current_weather,
        }
        for tool_call in tool_calls:
            print(f"Function: {tool_call.function.name}")
            print(f"Params:{tool_call.function.arguments}")
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
            )
            #   print(f"API: {function_response}")
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                }
            )

        second_response = client.chat.completions.create(
            model=model_name, messages=messages, stream=True
        )
        return second_response


if __name__ == "__main__":
    question = "What's the weather like in Paris"
    response = run_conversation(question)
    #   print(response)
    for chunk in response:
        # Skip empty chunks
        if not chunk.choices or not hasattr(chunk.choices[0], "delta"):
            continue

        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            print(delta.content, end="", flush=True)
