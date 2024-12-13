USER_DEMAND_REPHRASE = """
You are web scraper, you are given a user demand, and you need to rephrase the user demand into a more specific and clear phrase. Based on the user input and queried data schema, you need to make a clear declaration of the user demand and enhance the importance of the data schema.
Consider the following infos:

**User Input**:
{user_input}

**Data Schema**:
{data_schema}

**Response in chinese**
"""

USER_DEMAND_REPHRASE_V2 = """
You are an expert on analysis and dig into user demand. You are given a user demand and your top 1 priority is to make it clear, specific and quantifiable. User demand is always about aquire data, so you need to make it clear what data user want. That means you must list out what specific data and time span the user want based on the given current information.

**USER RAW INPUT**:
{user_input}

**DATA SCHEMA**:
{data_schema}

**RESPONSE**
"""

USER_DEMAND_REPHRASE_REGENERATE = """
You are an expert on analysis and dig into user demand. You are given a user demand and your top 1 priority is to make it clear, specific and quantifiable. Now your task is to refine the user demand details based on the new information that collected from the system observations.

**USER RAW INPUT**:
{user_input}

**USER PREVIOUS REPHRASE DETAILS**:
{user_rephrased_input}

**DATA SCHEMA**:
{data_schema}

**NEW SYSTEM OBSERVATIONS**:
{system_observations}

**RESPONSE**
"""


API_CALL_FILTER = """
You task is to analysis if this api call is relevant to the user demand.\n
If this api is necessary to statisfy the user data extraction or prerequisite of data extraction, return YES, otherwise return NO.
Here is the supplimentary infos:

**User Input**:
{user_rephrased_input}

**API Call**:
{api_event}

**Response**:
"""

API_FUNCTION_ANALYSIS = """
You are expert on api development, you are given a api call example, you need to figure out the api function. What this api can do, the possible meaning of each request parameters and what content the api response may contain.

**API Call Example**:
{api_call}

**Response**:
"""

API_FUNCTION_ANALYSIS_V2 = """
You are a backend developer, you are given a api call example, based on the given api call example, you need to figure out the api function. What this api can do, the possible meaning of each request parameters and what content the response body may contain.

**API CALL EXAMPLE**:
{api_call}

**RESPONSE**:
"""


PLANNING = """
You are a web scraper and mainly focused on solving data extraction demands with api calling. You are given several api call examples and you need to figure out a plan to get the data user want by using these api calls.

**TASK STATUS**:
{task_status}

**USER INPUT**:
{user_rephrased_input}

**API FUNCTION ANALYSIS**:
{api_function_analysis}

**HISTORY SUB-TASK INFO**:
{history_sub_task_completion}

Planning instructions:
1. First, you must have a clear knowledge of the api function based on the given API CALL EXAMPLES.
2. In order to meet the user's demand, you have to decide a serie of sub tasks to take to get things done. For each sub task, you have to choose the correct api call from the API CALL EXAMPLES and make a clear task goal and target. So that the following execution worker will have a clear clue on how to work on it. 
3. If some sub tasks are done and provided in the HISTORY SUB-TASK INFO, that means these tasks have been completed, ignore these tasks and do not include them in the following planning.
4. Some user demands can be satisfied by only one api request, only one thing you may do is to find the correct api calling method, for example, populate the correct request parameters. This parameter filling clue may exist in other api responses, so you may have to make an assumptions on some api and make a task to check out this clue.
5. Other user demands may need multiple api calls to get the final data. In one case, these api calls are parallel, what you have to do is call these api and conclude all the responses, to get the final data. In another case, these api calls are sequential, what you have to do is call these api one by one, and use the response of previous api call to populate the request parameters of the next api call.
6. Output the task planning in json format following this format instruction:
{format_instruction}\n
7. When make a plan, do not miss the details from USER INPUT. Details are those things about data target, time span, etc.

Think step by step. First, give you thinking progress, how to use the above api to meet the user demand. and then output the task planning in markdown json format.

ONLY MAKE ONE SUB TASK PLAN, DO NOT MAKE MULTIPLE SUB TASK PLANS.

WHEN MAKE SUB TASK PLAN, YOU MUST CONSIDER WHICH PART IS NOT PRESENT AND MUST DO FIRST.
 
**Response in chinese**:
"""


SUB_TASK_API_CALL_TRAIL = """
You are a api call tester, you got a task to test the api call based on a given api call example. You need figure out in order to meet the task target, how to manipulate the api call request and show the api response. Now you asked to generate code to do this. The following information is for you background knowledge:

**USER RAW INPUT**:
{user_rephrased_input}

**SUB TASK**:
{task_target}

**API Call Example**:
{api_call}

**API FUNCTION ANALYSIS**:
{api_call_function}

Based on these context, generate the `api_call_test() -> dict()` function that:
1. Modify the api call request based on the task target, make a real api call and return the response.
2. If you cannot figure out how to do it with this context, you can return simple code  ```python\nprint("No clue on task.")\n```.

Your code should be well-commented, explaining the reasoning behind key decisions and any potential areas for improvement or customization.

Use only the following pre-imported libraries:
- requests
- re
- json
- time

**Output ONLY the Python code of the api_call_test function, WITHOUT ANY IMPORTS OR ADDITIONAL TEXT.**
**ONLY output the raw request response body, do not try the extract data from response in any format.**
**In your code do not include backticks.**
**Response**:
"""

SUB_TASK_FAILED_ANALYSIS = """
Your task is to analyze the reason the sub task failed. Based on the task details and the api call code and code execution result, you need to figure out the reason why demand is not satisfied.

**SUB TASK**:
{task_target}

**API Call Code**:
{api_call_code}

**API Call Code Execution Result**:
{api_call_code_execution_result}

**API CALL EXAMPLE**:
{api_call}

Please analyze why the result is not desired.
Provide your analysis and suggestions for fixing the error. DO NOT generate any code in your response.
TIPS:
1. Under the circumstances that you have to make a recursive logic to get the final result, you need to emphasize this point.
2. When analyze the error about time span, the data time span user request has to be included in the response.

**Response**:
"""

SUB_TASK_API_CALL_TRAIL_REGENERATE = """
You are a api call tester, you have to regenerate a new version of api calling code based on the previous code generation, code execution result error analysis and other context. You need figure out in order to meet the task target, how to manipulate the api call request and show the api response. Now you asked to regenerate code to do this.


**SUB TASK**:
{task_target}

**PREVIOUS ERROR API CALL CODE**:
{api_call_code_gen}

**PREVIOUS ERROR API CALL CODE EXECUTION RESULT**:
{api_call_code_execution_result}

**PREVIOUS ERROR API CALL ANALYSIS**:
{api_call_code_error_analysis}

Based on these context, generate the `api_call_test() -> dict()` function that:
1. Modify the api call request based on the task target, make a real api call and return the response.
2. If you cannot figure out how to do it with this context, you can return simple code  ```python\nprint("No clue on task.")\n```.

Your code should be well-commented, explaining the reasoning behind key decisions and any potential areas for improvement or customization.

Use only the following pre-imported libraries:
- requests
- re
- json
- time

**Output ONLY the Python code of the api_call_test function, WITHOUT ANY IMPORTS OR ADDITIONAL TEXT.**
**ONLY output the raw request response body, do not try the extract data from response in any format.**
**In your code do not include backticks.**
**Response**:
"""


API_EXECUTION_ERROR_ANALYSIS = """
The current code has encountered an execution error. Here are the details:

**Current Code**:
```python
{generated_code}
```
**Execution Error**:
{errors}


Please analyze the execution error and suggest a fix. Focus only on correcting the execution issue while ensuring the code still meets the original requirements and maintains correct syntax.
Provide your analysis and suggestions for fixing the error. DO NOT generate any code in your response.
"""

API_CALL_UPDATE = """
Given the original api call example and the new version of generated api calling code and code execution result, you need to update the api call example based on the new version of generated api calling code and code execution result.

**Original API Call Example**:
{api_call}

**New API Call Code**:
{api_call_code}

**New API Call Code Execution Result**:
{api_call_code_execution_result}

Output in json format.

**Response**:
"""


API_OUTPUT_EVALUATION = """
Given the task description and the api call result, you need to evaluate if the api call result satisfies the task goal. If it does, return "YES", otherwise return "NO".

**Task Description**:
{task_target}

**API Call Result**:
{api_call_result}

**BE CAREFUL: sometimes the api call result is not the final result, it may be a intermediate result, so you need to make a judgement on the api call result whether it is the final result.**
TIPS:
1. Under the circumstances that you have to make a recursive logic to get the final result, you need to emphasize this point.
2. When analyze the error about time span, the data time span user request has to be included in the response.


**Response**:
"""

OVERALL_EVALUATION = """
You task is to evaluate the current api testing results whether meet the user demand. Review the history task execution result，find out the api response is adequate to get the data user want. If api call results meet the user demand, return "YES", otherwise return "NO".

**USER INPUT**:
{user_rephrased_input}

**History Task Execution Result**:
{history_sub_task_completion}

**Response**:
"""


# 这里分析后的调用链可能是一个并行 trails，后续存在多个结果汇总后分析拿到总结过。也可能是一个串行 trails，串行的拿到中间结果并进行分析。也可能是一个混合交叉的逻辑。这会导致后续的代码生成任务是多个子任务的并行或者串行。workflow/agentic 要在实现层面的结构上支持相关逻辑。
# TODO 在接下来的实现中，code gen 会在一个函数中打包所有调用逻辑，如果调用链不复杂的情况下应该是扣的，但是复杂问题就要进一步想想解决方案了。


# 根据策略，将任务分解为多个子任务，根据后续的子任务结果做进一步规划。
STRATEGY_TASK_DECOMPOSITION = """
"""


# 写代码部分的任务还是一个明确任务的闭环逻辑，真实情况应该在外部做好任务拆机和规划。conquer-divide-conquer
API_CALL_CODE_GENERATION = """
Your are a web scraping developer, now you are given several api calling examples and possible calling strategy. What you have to do is based one the strategy, try to write down code to call api and get response. And this response that may be useful for the downstream tasks.

**User Input**:
{user_rephrased_input}

**API Calling Examples**:
{api_call_candidates}

**API Calling Strategy**:
{api_calling_strategy}

Your code should be well-commented, explaining the reasoning behind key decisions and any potential areas for improvement or customization.

**Output ONLY the Python code of the extract_data function, WITHOUT ANY IMPORTS OR ADDITIONAL TEXT.**
In your code do not include backticks.

**Response**:
"""

API_CALL_CODE_SYNTAX_CHECK = """

"""

API_CALL_CODE_SYNTAX_FIX = """

"""

API_CALL_CODE_EXECUTION_CHECK = """
"""

API_CALL_CODE_EXECUTION_FIX = """
"""

API_CALL_CODE_OUTPUT_CHECK = """
"""
