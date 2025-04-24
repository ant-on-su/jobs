# Job Vacancy Extractor

A CLI tool that extracts job vacancy information from webpages and uses AI to structure the data into a standardized JSON format.

## Features

- Fetches content from any webpage URL
- Extracts readable text from HTML
- Uses OpenAI and LangChain to analyze job postings
- Outputs structured job information in JSON format
- Extracts the following fields:
  - Company or project details
  - Vacancy description
  - Vacancy tasks and responsibilities
  - Candidate requirements
  - Vacancy benefits

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd jobs
   ```

2. Install the package using uv:
   ```bash
   uv pip install -e .
   ```

3. Set up your OpenAI API key:
   - Copy the `.env.example` file to `.env`
   - Replace the placeholder with your actual OpenAI API key
   ```bash
   cp .env.example .env
   # Edit .env with your text editor
   ```

## Usage

### Basic Usage

```bash
fetch-webpage https://example.com/job-posting
```

### Save Output to a File

```bash
fetch-webpage https://example.com/job-posting -o job_info.json
```

### Get Raw Text Without LLM Processing

```bash
fetch-webpage https://example.com/job-posting --raw
```

### Provide API Key Directly

```bash
fetch-webpage https://example.com/job-posting --api-key your_api_key_here
```

### Using Different LLM Models

The tool supports various LLM backends through LiteLLM. You can specify the model to use with the `--model` option:

```bash
# Using OpenAI models (default)
fetch-webpage https://example.com/job-posting --model gpt-3.5-turbo

# Using Anthropic Claude
fetch-webpage https://example.com/job-posting --model claude-3-sonnet-20240229
```

### Using Local LLM Models

You can use locally hosted models by specifying the API base URL:

```bash
# Using Ollama with Llama2
fetch-webpage https://example.com/job-posting --api-base http://localhost:11434 --model ollama/llama2

# Using a local LLM Studio server
fetch-webpage https://example.com/job-posting --api-base http://localhost:1234/v1 --model local-model
```

### Get Help

```bash
fetch-webpage --help
```

## Output Format

The tool outputs a JSON object with the following structure:

```json
{
  "company_or_project_details": "Information about the company...",
  "vacancy_description": "General description of the job...",
  "vacancy_tasks": "Specific tasks and responsibilities...",
  "candidate_requirements": "Requirements and qualifications...",
  "vacancy_benefits": "Benefits and perks offered..."
}
```

## Dependencies

- requests: For HTTP requests
- beautifulsoup4: For HTML parsing
- click: For CLI interface
- langchain: For LLM integration
- litellm: For multi-provider LLM support
- pydantic: For data validation
- python-dotenv: For environment variable management

## Supported LLM Providers

Through LiteLLM, this tool supports over 100 different LLM providers, including:

- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Google (Gemini)
- Cohere
- Local models via:
  - Ollama
  - LM Studio
  - LocalAI
  - And more

For a complete list of supported providers, see the [LiteLLM documentation](https://docs.litellm.ai/docs/providers).
