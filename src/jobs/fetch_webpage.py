#!/usr/bin/env python3
"""
A tool to extract job vacancy information from webpages.
"""

import sys
import os
import json
import requests
from bs4 import BeautifulSoup
import click
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
import litellm
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun


# Load environment variables from .env file
load_dotenv()


class JobVacancy(BaseModel):
    """Schema for job vacancy information."""
    company_or_project_details: str = Field(
        description="Information about the company or project offering the job"
    )
    vacancy_description: str = Field(
        description="General description of the job vacancy"
    )
    vacancy_tasks: str = Field(
        description="Specific tasks and responsibilities for the role"
    )
    candidate_requirements: str = Field(
        description="Requirements and qualifications needed from candidates"
    )
    vacancy_benefits: str = Field(
        description="Benefits and perks offered with the position"
    )


class LiteLLMWrapper(LLM):
    """Wrapper around LiteLLM to use with LangChain."""
    
    model_name: str = "gpt-3.5-turbo"
    """Model name to use."""
    
    temperature: float = 0.0
    """LLM temperature."""
    
    api_base: Optional[str] = None
    """Base URL for API."""
    
    api_key: Optional[str] = None
    """API Key."""
    
    @property
    def _llm_type(self) -> str:
        return "litellm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Call the LiteLLM API."""
        messages = [{"role": "user", "content": prompt}]
        
        # Prepare parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        # Add API base if provided
        if self.api_base:
            params["api_base"] = self.api_base
        
        # Add API key if provided
        if self.api_key:
            params["api_key"] = self.api_key
        
        # Call LiteLLM
        response = litellm.completion(**params)
        
        return response.choices[0].message.content


def extract_text_from_html(html_content):
    """
    Extract readable text from HTML content using BeautifulSoup.
    
    Args:
        html_content (str): HTML content as string
        
    Returns:
        str: Extracted text from the HTML
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
        script_or_style.extract()
    
    # Get text
    text = soup.get_text()
    
    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    
    # Remove blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text


def extract_job_info(text: str, model: str = "gpt-3.5-turbo", api_base: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract job vacancy information from text using LangChain and LiteLLM.
    
    Args:
        text (str): Text content from webpage
        model (str): Model name to use
        api_base (str, optional): Base URL for API if using a local model
        
    Returns:
        Dict[str, Any]: Extracted job information in JSON format
    """
    # Check if API key is set for remote models
    if not api_base and "OPENAI_API_KEY" not in os.environ and not model.startswith("ollama/"):
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in your environment or in a .env file, "
            "or use a local model with --api-base."
        )
    
    # Initialize the LLM
    llm = LiteLLMWrapper(
        model_name=model,
        temperature=0,
        api_base=api_base,
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Set up the parser
    parser = PydanticOutputParser(pydantic_object=JobVacancy)
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert at analyzing job postings and extracting structured information.
        
        Please analyze the following text from a job posting webpage and extract the relevant information.
        You need to extract the text word to word, exactly as it appears in the posting.
        
        Search in the text and extract the sections or paragraphs that match the following categories:
        1. Company or project details: a more general description of the company or (and) the focus of the project
        2. Vacancy description: a more general description of what the vacancy is about
        3. Vacancy tasks and responsibilities: specific tasks and responsibilities for the role, a list of tasks
        4. Candidate requirements and qualifications: list of requirements and qualifications needed from candidates
        5. Vacancy benefits and perks: list of benefits and perks offered with the position
        
        If any section is not found in the text, please indicate with "Not specified in the posting".
        
        Text from webpage:
        {text}
        
        {format_instructions}
        """
    )
    
    # Set up the chain
    chain = prompt | llm | parser
    
    # Run the chain
    try:
        result = chain.invoke({
            "text": text,
            "format_instructions": parser.get_format_instructions(),
        })
        return result.dict()
    except Exception as e:
        raise Exception(f"Error extracting job information: {e}")


@click.command()
@click.argument('url')
@click.option('--output', '-o', type=click.Path(), help='Save output to a file instead of printing to stdout')
@click.option('--raw', is_flag=True, help='Output raw text without LLM processing')
@click.option('--api-key', help='API key for remote models (overrides environment variable)')
@click.option('--model', default="gpt-3.5-turbo", help='Model to use (default: gpt-3.5-turbo)')
@click.option('--api-base', help='Base URL for API if using a local model')
def fetch_webpage(url, output, raw, api_key, model, api_base):
    """
    Fetch job vacancy information from a webpage.
    
    URL: The URL of the webpage to fetch
    """
    try:
        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Add user-agent to avoid being blocked by some websites
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the webpage
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Extract text from HTML
        text = extract_text_from_html(response.text)
        
        # Process with LLM if not raw
        if not raw:
            try:
                result = extract_job_info(text, model=model, api_base=api_base)
                output_text = json.dumps(result, indent=2, ensure_ascii=False)
            except ValueError as e:
                if "OPENAI_API_KEY" in str(e):
                    click.echo(f"Error: {e}", err=True)
                    click.echo("You can provide an API key with --api-key option", err=True)
                    sys.exit(1)
                else:
                    raise
        else:
            output_text = text
        
        # Output the result
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(output_text)
            click.echo(f"Output saved to {output}")
        else:
            click.echo(output_text)
            
    except requests.exceptions.RequestException as e:
        click.echo(f"Error fetching webpage: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"An error occurred: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    fetch_webpage()