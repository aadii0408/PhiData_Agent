from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Create an instance of the Groq model
model_instance = Groq(id="llama-3.3-70b-versatile")

# Create an instance of Agent with the model
agent = Agent(model=model_instance)

# Use the agent to print a response
agent.print_response("Create a 4-line poem about a llama")
