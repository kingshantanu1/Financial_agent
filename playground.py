import openai
from phi.agent import Agent
#import phi.api
from phi.model.openai import OpenAIChat 
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import phi
from phi.playground import Playground, serve_playground_app
from phi.model.groq import Groq


load_dotenv()

PHI_API_KEY="phi-A-S5J4i1cJoRkFCVcAUISCnBV072RJU1g8T3Fmcihe4"
#phi.api = os.getenv("PHI_API_KEY")

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

app = Playground(
    agents=[websearch_agent, financial_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)