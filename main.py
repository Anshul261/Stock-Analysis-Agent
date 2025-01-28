import os
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.agent_toolkits import FileManagementToolkit
from tools.yahoo_finance import yahoo_finance_tool
import os
from langchain_openai import AzureChatOpenAI

load_dotenv()

file_management_fools = FileManagementToolkit(
    root_dir=str("./"),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()
read_tool, write_tool, list_tool = file_management_fools

# Set up the agent
def setup_agent():
    tools = [read_tool, write_tool, list_tool, yahoo_finance_tool]
    
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    llm = AzureChatOpenAI(
    deployment_name="gpt-4",
    model_name="gpt-4",
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_version=os.getenv('OPENAI_API_VERSION'),
    openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    )
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# main code execution
def main():

    ticker_symbol = "NVDA"
    
    agent_executor = setup_agent()

    prompt = f"""
        description:
        first for a given ticker: {ticker_symbol} fetch the data using the yahoo_finance_tool 
        then get warren buffett's investment principles by opening
        the file warren_buffet_investement_principles.txt
        then you provide a comprehensive analysis of the ticker
        write short and concise the pros why warren buffett would invest in
        this company and the cons why he wouldn't
        then summarize the company evaluation and provide a recommendation
        then you give a warren buffet buy recomendation from 0 to 10 (10 is best)
        finally write the result to the file stock_analysis.txt
        expected_output:
        - Pros why Warren Buffett would invest in this company
        - Cons why Warren Buffett wouldn't invest in this company
        - Company Evaluation Summary
        - Warren Buffet Buy Recommendation
    """
    
    # Example invocation will be converted to api for app use
    result = agent_executor.invoke({
        "input": prompt
    })
    
    print("Result:", result)

if __name__ == "__main__":
    main()