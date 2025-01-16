from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

from phi.playground import Playground, serve_playground_app

web_agent = Agent(
    name="Web Agent",
    role="Get data from internet",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    show_tool_calls=True,
    markdown=True,
    instructions=["Always include sources."],
    debug_mode=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True)],
    show_tool_calls=True,
    markdown=True,
    instructions=["Use tables to display data"],
    debug_mode=True,
)

app = Playground(agents=[web_agent, finance_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
