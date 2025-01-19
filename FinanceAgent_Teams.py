from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
from textblob import TextBlob

load_dotenv()


def analyze_sentiment(text):
    """Analyze sentiment of the text using TextBlob."""
    sentiment = TextBlob(text).sentiment
    return {
        "polarity": sentiment.polarity,
        "subjectivity": sentiment.subjectivity,
    }


def simple_stock_trend(stock_data):
    """Predict stock trend: up or down based on the last two days' closing prices."""
    if stock_data[-1] > stock_data[-2]:
        return "Up", "Confidence: High"
    elif stock_data[-1] < stock_data[-2]:
        return "Down", "Confidence: High"
    else:
        return "Stable", "Confidence: Medium"


# Define Web Agent for News and Finance Agent for Stock Data
web_agent = Agent(
    name="Web Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)
    ],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Define the main agent team combining both web and finance agents
agent_team = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Example Response to Summarize Analyst Recommendations and News with Sentiment
agent_team.print_response(
    "Summarize analyst recommendations and share the latest news for NVDA", stream=True
)

# print("Agent Response:", response)


# Assume the agent pulls some news for NVDA, this is just a placeholder for demonstration
latest_news = [
    "NVIDIA's stock shows strong performance amid growing demand for GPUs.",
    "NVIDIA stock hits record high due to partnerships with AI startups.",
    "Market reacts positively to NVIDIA's new product release.",
]

# Sentiment analysis for the latest news
sentiments = [analyze_sentiment(news) for news in latest_news]

# Display the sentiment analysis alongside the news
print("\nSentiment Analysis of Latest News:")
for i, news in enumerate(latest_news):
    sentiment = sentiments[i]
    print(f"News: {news}")
    print(
        f"Polarity: {sentiment['polarity']}, Subjectivity: {sentiment['subjectivity']}\n"
    )

# Example stock data (historical closing prices for NVDA, placeholder)
nvda_stock_data = [230, 232, 235, 240, 238]

# Predict stock trend
trend, confidence = simple_stock_trend(nvda_stock_data)

# Display the trend prediction
print(f"Stock Trend Prediction for NVDA: {trend} (Confidence: {confidence})\n")

# Show Analyst Recommendations (example)
print("Analyst Recommendations for NVDA:")
# Assuming agent pulls recommendation data (this is a placeholder example)
recommendations = {
    "Buy": 15,
    "Hold": 5,
    "Sell": 2,
}
for recommendation, count in recommendations.items():
    print(f"{recommendation}: {count} analysts")
