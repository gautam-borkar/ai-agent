from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

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

agent_team = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_agent, finance_agent],
    show_tool_calls=True,
    markdown=True,
    instructions=["Always include sources.", "Use tables to display data"],
    debug_mode=True,
)

agent_team.print_response(
    "Summarize analyst recomendations and share latest news for Nvdia", stream=True
)
