from dotenv import load_dotenv
import os

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from todoist_api_python.api import TodoistAPI

load_dotenv()

todoist_api_key = os.getenv('TODOIST_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

todoist = TodoistAPI(todoist_api_key)

@tool
def add_task(task, desc=None):
    """Add a new task to the user's task list. Use this when the user wants to add or create a new task."""
    todoist.add_task(
        content=task,
        description=desc
    )

tools = [add_task]

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=gemini_api_key,
    temperature=0.3
)

system_prompt = "You are a helpful assistant. You will help the user to add tasks. Your answers should be offensive."
user_input = "Add task get a new tire with description get a new tire for my car"

prompt = ChatPromptTemplate([
    ("system", system_prompt),
    ("user", user_input),
    MessagesPlaceholder("agent_scratchpad")
])

# chain = prompt | llm | StrOutputParser()
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# response = chain.invoke({
#     "input":user_input
# })

response = agent_executor.invoke({
    "input": user_input
})

print(response['output'])