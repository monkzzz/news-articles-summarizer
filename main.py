import requests
from newspaper import Article

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import field_validator
from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv
load_dotenv()

class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted list summary of the article")

    # validating whether the generated summary has at least three lines
    @field_validator('summary')
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated summary has less than three bullet points!")
        return list_of_lines

# set up output parser
parser = PydanticOutputParser(pydantic_object=ArticleSummary)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

article_urls = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"

session = requests.Session()

try:
    response = session.get(article_urls, headers=headers, timeout=10)

    if response.status_code == 200:
        article = Article(article_urls)
        article.download()
        article.parse()
        
        print("")
        print("Article: ")
        print("")
        print(f"Title: {article.title}")
        print(" ")
        print(f"Text: {article.text}")
        print(" ")

        # we get the article data from the scraping part
        article_title = article.title
        article_text = article.text

        # create prompt template
        # notice that we are specifying the "partial_variables" parameter
        template = """
        You are a very good assistant that summarizes online articles.

        Here's the article you want to summarize.

        ==================
        Title: {article_title}

        {article_text}
        ==================

        {format_instructions}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["article_title", "article_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # Format the prompt using the article title and text obtained from scraping
        formatted_prompt = prompt.format_prompt(article_title=article_title, article_text=article_text)

        # instantiate model class
        model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)

        # Use the model to generate a summary
        output = model.invoke(formatted_prompt.to_string())
        print("Output:")
        print(output)
        print(" ")

        # Parse the output into the Pydantic model
        parsed_output = parser.parse(output)
        print("Parsed Output:")
        print(parsed_output)
        print(" ")

    else:
        print(f"Failed to fetch article at {article_urls}")

except Exception as e:
    print(f"Error occurred while fetching article at {article_urls}: {e}")