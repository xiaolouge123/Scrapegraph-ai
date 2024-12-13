"""
Generate code API prompts helper
"""

TEMPLATE_INIT_CODE_API_GENERATION = """
You are a web scraper expert. Your task is to write code to do web scraping given the following user's request and desired JSON Output Schema.
**Task**: Create a Python function named `extract_data() -> dict()` that extracts relevant information by actually calling the given API calls and returns it in a desired JSON Output Schema.
Understand the request payload to understand the parameters populated in the API calls.
Understand the structure of the data returned by the API calls. Response data body is only for reference, DO NOT copy the data directly.

**User's Request**:
{user_input}

**Desired JSON Output Schema**:
```json
{json_schema}
```

**API Analysis**:
{api_analysis}

**API Calls**:
{candidates_api_events}

Based on the above analyses, generate the `extract_data() -> dict()` function that:
1. Efficiently extracts the required data from the given API calls.
2. Processes and structures the data according to the specified JSON schema.
3. Returns the structured data as a dictionary, with the key `data`.
4. DO NOT simulate the API calls in code, just write the real code to call the API directly.

Your code should be well-commented, explaining the reasoning behind key decisions and any potential areas for improvement or customization.

**Output ONLY the Python code of the extract_data function, WITHOUT ANY IMPORTS OR ADDITIONAL TEXT.**
In your code do not include backticks.

**Response**:
"""


TEMPLATE_API_EXECUTION_ANALYSIS = """
The current code has encountered an execution error. Here are the details:

**Current Code**:
```python
{generated_code}
```

**Execution Error**:
{errors}

**API Analysis**:
{api_analysis}

**API Calls**:
{candidates_api_events}

Please analyze the execution error and suggest a fix. Focus only on correcting the execution issue while ensuring the code still meets the original requirements and maintains correct syntax.
Provide your analysis and suggestions for fixing the error. DO NOT generate any code in your response.
"""

API_REFERENCE_DATA_GENERATION = """
You are a web scraper expert. Your task is to satisfy the user's data extraction request.
Given the following information:

**User's Request**: {user_input}
**Desired JSON Output Schema**: {json_schema}
**API Calls**: {candidates_api_events}

Your task is to generate the reference data that satisfies the user's request.
1. Output the reference data in JSON format.
2. DO NOT include any additional text or comments in your response.

**Response**:
"""
