from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv

load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")

#Creating Web search agent
websearch_agent = Agent(
    name="WebSearchAgent",
    description="An agent that can search the web for information.",
    tools=[DuckDuckGo()],
    model=Groq(id="llama3-70b-8192"),
    role="Search the web for information and answer questions based on the search results.",
    instructions=["Always use sources"],
    show_tool_calls=True,
    markdown=True
)


#Creatring Financial agent
financial_agent = Agent(
    name="Financial Agent",
    model=Groq(id="llama3-70b-8192"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)
        ],
    instructions=[
        "Use tables to display data."
    ],
    show_tool_calls=True,
    markdown=True
)


multi_model_agent = Agent(
    team=[websearch_agent, financial_agent],
    instructions=["Always use sources", "use tables to display data"],
    show_tool_calls=True,
    markdown=True
)

multi_model_agent.print_response("Summarize Analyst Reccomendation and share the latest news for NVIDIA", stream=True)