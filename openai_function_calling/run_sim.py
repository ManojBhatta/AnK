import subprocess
import os
import json
from openai import AzureOpenAI    
from dotenv import load_dotenv

def run_sim(tg, ta, x1, x2, x3, t):
    """
    Runs the Julia simulation for 3D viscoelastic bending of glass using TemperFEM.

    Parameters:
        tg (float): Initial glass temperature.
        ta (float): Ambient air temperature.
        x1 (float): Width of the glass along dimension 1.
        x2 (float): Width of the glass along dimension 2.
        x3 (float): Width of the glass along dimension 3.
        t (float): Thickness of the glass.

    This function calls a Julia script with the specified parameters using subprocess.
    """
    
    try:
        result = subprocess.run([
            "julia", "--threads=auto", 
            "TemperFEM/scripts/pressnet_viscoelastic_bending_3D.jl", 
            str(tg), str(ta), str(x1), str(x2), str(x3),  str(t)
        ], capture_output=True, text=True, check=True)
        
        return {
            "status": "success",
            "message": "Simulation completed successfully",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "parameters": {
                "glass_temp": tg, "ambient_temp": ta, "x1": x1, "x2": x2, "x3": x3, "thickness": t
            }
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Simulation failed with return code {e.returncode}",
            "stdout": e.stdout,
            "stderr": e.stderr,
            "parameters": {
                "glass_temp": tg, "ambient_temp": ta, "x1": x1, "x2": x2, "x3": x3, "thickness": t
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "parameters": {
                "glass_temp": tg, "ambient_temp": ta, "x1": x1, "x2": x2, "x3": x3, "thickness": t
            }
        }

# Load environment variables from .env file
load_dotenv()    

# Create Azure OpenAI client
client = AzureOpenAI()

model_name = os.getenv("AZURE_DEPLOYMENT_NAME")


def run_sim_from_llm(prompt):
    """
    Runs a conversation with the LLM that can call the Julia simulation function.
    
    Parameters:
        prompt (str): The user prompt requesting a simulation
        
    Returns:
        dict or str: Function response if tool was called, otherwise original LLM response
    """
    messages = [{"role": "user", "content": prompt}]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_sim",
                "description": "Runs the Julia simulation for 3D viscoelastic bending of glass using TemperFEM",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tg": {
                            "type": "number",
                            "description": "Initial glass temperature in Celsius",
                        },
                        "ta": {
                            "type": "number",
                            "description": "Ambient air temperature in Celsius",
                        },
                        "x1": {
                            "type": "number",
                            "description": "Width of the glass along dimension 1 in meters ",
                        },
                        "x2": {
                            "type": "number", 
                            "description": "Width of the glass along dimension 2 in meters ",
                        },
                        "x3": {
                            "type": "number",
                            "description": "Width of the glass along dimension 3 in meters ",
                        },
                        "t": {
                            "type": "number",
                            "description": "Thickness of the glass in meters ",
                        },
                    },
                    "required": ["tg", "ta", "x1", "x2", "x3", "t"],
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
        available_functions = {
            "run_sim": run_sim,
        }
        
        for tool_call in tool_calls:
            print(f"Function: {tool_call.function.name}")
            print(f"Params: {tool_call.function.arguments}")
            
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
            # Call the function (simulation runs here)
            function_response = function_to_call(
                tg=function_args.get("tg"),
                ta=function_args.get("ta"),
                x1=function_args.get("x1"),
                x2=function_args.get("x2"),
                x3=function_args.get("x3"),
                t=function_args.get("t"),
            )
            
            print("Simulation is running...")
            return function_response
    
    # If no tool calls, return the original response
    return response_message.content


if __name__ == "__main__":
    # Example usage
    question = """Run a Julia simulation with the following parameters:
    - Initial glass temperature: 600 degrees Celsius
    - Ambient temperature: 450 degrees Celsius
    - Width 1 (x1): 0.1 meters
    - Width 2 (x2): 0.2 meters  
    - Width 3 (x3): 0.15 meters
    - Glass thickness: 0.003 meters
    """
    
    response = run_sim_from_llm(question)
    print(response)