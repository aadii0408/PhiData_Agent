from phi.agent import Agent
from dotenv import load_dotenv
from phi.model.groq import Groq


load_dotenv()

# Create an instance of the Finance model
model_instance = Groq(id="llama-3.3-70b-versatile")

# Create an instance of Agent with the model
agent = Agent(model=model_instance)

# Use the agent to print a response
agent.print_response(
    "Summerize and compare the financials of Apple and Microsoft. Also give recommendations. Show in tables.",
    stream="True",
)
