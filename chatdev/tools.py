import os
from dotenv import load_dotenv
import time
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage

load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()

# 1. Tool for search

def search(query):
    url = "https://google.serper.dev/search"
 
    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text


# 2. Tool for scraping

def scrape_website(objective: str, url: str, browserless_api_key: str):
    # Scrape a website and perform summarization if the content is too large.
    # - objective: The original objective & task provided by the user.
    # - url: The URL of the website to be scraped.
    # - browserless_api_key: API key for browserless.io service.

    try:
        print("Scraping website...")
        # Define the headers for the request
        headers = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
        }

        # Define the data to be sent in the request
        data = {
            "url": url
        }

        # Convert Python object to JSON string
        data_json = json.dumps(data)

        # Send the POST request
        post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
        response = requests.post(post_url, headers=headers, data=data_json)
        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes

        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()

        if len(text) > 0:
            if len(text) > 10000:
                # Perform summarization if content is too large
                output = summary(objective, text)
                return output
            else:
                return text
        else:
            print("Scraped content is empty.")
            return ""  # Return an empty string if content is empty

    except requests.exceptions.RequestException as req_ex:
        print(f"HTTP request failed: {req_ex}")
        return ""  # Return an empty string on request failure

    except Exception as e:
        print(f"Error during web scraping: {e}")
        return ""  # Return an empty string on other errors

# Example usage:
# scraped_content = scrape_website("Objective", "https://example.com", "your_browserless_api_key_here")



def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ Prioritize recent results and search for diverse sources
            3/ If there are url of relevant links & articles, you will scrape it to gather more information
            4/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iterations
            5/ You should not make things up, you should only write facts & data that you have gathered
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            7/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)


agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Research tools")
    parser.add_argument("--search", help="Search the web using a given query", type=str)
    parser.add_argument("--scrape", help="Scrape a given URL", action="store_true")
    parser.add_argument("--objective", help="Objective for scraping", type=str)
    parser.add_argument("--url", help="URL to scrape", type=str)
    
    args = parser.parse_args()

    if args.search:
        print(search(args.search))
    elif args.scrape and args.objective and args.url:
        print(scrape_website(args.objective, args.url))
    else:
        print("Please provide valid arguments.")
