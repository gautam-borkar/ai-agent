from dotenv import load_dotenv

from phi.agent import Agent

from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools

load_dotenv()


def get_company_symbol(company: str) -> str:
    """
    Use this function to get the company symbol for a given company.

    company: str - The name of the company.

    returns: str - The stock symbol for the given company.
    """
    symbol = {"Phidata": "MSFT", "Tesla": "TSLA", "Google": "GOOGL", "Apple": "AAPL"}
    return symbol.get(company)


agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(
            stock_price=True, stock_fundamentals=True, analyst_recommendations=True
        ),
        get_company_symbol,
    ],
    show_tool_calls=True,
    markdown=True,
    instructions=[
        "Use tables to display data.",
        "If company's stock symbols is not found then please use get_company_symbol tool.",
    ],
    debug_mode=True,
)

agent.print_response(
    "Summarize and compare analyst recomendations and fundamentals for Tesla and Phidata"
)
