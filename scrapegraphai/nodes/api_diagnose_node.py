import ast
import json
import re
import sys
import time
import requests
from io import StringIO
from pydantic import BaseModel, Field
from tqdm import tqdm
from requests.exceptions import Timeout
from urllib.parse import urlparse
from typing import List, Dict
from .base_node import BaseNode
from ..utils import (
    transform_schema,
    syntax_focused_analysis,
    syntax_focused_code_generation,
    execution_focused_code_generation,
    validation_focused_analysis,
    validation_focused_code_generation,
    semantic_focused_analysis,
    semantic_focused_code_generation,
    are_content_equal,
)
from ..utils.output_parser import get_pydantic_output_parser
from ..utils import transform_schema
from ..prompts.api_diagnose_node_prompts import (
    USER_DEMAND_REPHRASE,
    USER_DEMAND_REPHRASE_V2,
    USER_DEMAND_REPHRASE_REGENERATE,
    API_FUNCTION_ANALYSIS,
    API_FUNCTION_ANALYSIS_V2,
    PLANNING,
    SUB_TASK_API_CALL_TRAIL,
    SUB_TASK_API_CALL_TRAIL_REGENERATE,
    SUB_TASK_FAILED_ANALYSIS,
    API_EXECUTION_ERROR_ANALYSIS,
    API_OUTPUT_EVALUATION,
    OVERALL_EVALUATION,
    API_CALL_UPDATE,
)
from langchain.prompts import PromptTemplate
from langchain.output_parsers import BooleanOutputParser
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel


import re


def extract_code(code: str) -> str:
    """
    Module for extracting code
    """
    code = code.strip()
    if code.startswith("def"):
        return code
    pattern = r"```(?:python)?[ ]?\n(.*?)```"

    match = re.search(pattern, code, re.DOTALL)

    return match.group(1) if match else code


def extract_api_call_example(code: str) -> str:
    pattern = r"```(?:json)?[ ]?\n(.*?)```"
    match = re.search(pattern, code, re.DOTALL)
    return match.group(1).replace("\n", "") if match else code


class APIDiagnoseNode(BaseNode):
    """
    A node responsible for diagnosing the API events and locating the api calling strategy.
    This node will analysis the api event and find out the api using strategy.
    If the single api is enough to get the data, this node will find out the correct api calling method.
    If the single api is not enough to get the data, this node will find out the combination of api calls to locate the data source.

    这个节点的主要逻辑是要能探索出api的调用方法，来满足用户的需求。能够根据需求判断需要什么样的 api 能够满足需求。这里存在一个 trail and error 的过程，来最终判定 api 调用方法。

    所以简化成如下流程节点
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Dict | None = None,
        node_name: str = "APIDiagnoseNode",
    ):
        super().__init__(node_name, "node", input, output, 3, node_config)

        self.llm_model = node_config["llm_model"]
        self.verbose = node_config.get("verbose", False)
        self.timeout = node_config.get("timeout", 120)

        self.max_iterations = node_config.get(
            "max_iterations",
            {
                "api_reasoning": 3,  # 外部的 api 调用策略分析大环设定 3 次
                "api_coding": 3,  # 内部的 api 调用代码生成、语法检查、执行检查、输出验证、输出结果检查 各设定 3 次
                "outer_loop": 5,  # 外部循环次数
                "inner_loop": 5,  # 内部循环次数
                "syntax": 3,
                "execution": 3,
            },
        )

    def invoke_with_timeout(self, chain, inputs, timeout):
        """Helper method to invoke chain with timeout"""
        try:
            start_time = time.time()
            response = chain.invoke(inputs)
            if time.time() - start_time > timeout:
                raise Timeout(f"Response took longer than {timeout} seconds")
            return response
        except Timeout as e:
            self.logger.error(f"Timeout error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error during chain execution: {str(e)}")
            raise

    def execute(self, state: dict) -> dict:
        input_keys = self.get_input_keys(state)
        input_data = [state[key] for key in input_keys]
        url = input_data[0]
        user_prompt = input_data[1]
        api_events = input_data[2]

        if self.node_config.get("schema", None) is not None:
            output_parser = get_pydantic_output_parser(self.node_config["schema"])
            format_instructions = output_parser.get_format_instructions()
            simplefied_schema = str(
                transform_schema(self.node_config["schema"].schema())
            )

        reasoning_state = {
            "url": url,
            "api_events": api_events,
            "user_prompt": user_prompt,
            "schema": simplefied_schema,
            "format_instructions": format_instructions,
            "errors": {"syntax": [], "execution": [], "validation": []},
            "api_calling_strategy": [],
            "api_reasoning_iteration": 0,
            "api_coding_iteration": 0,
            "outer_loop_iteration": 0,
            "inner_loop_iteration": 0,
            "satisfied": False,
            "history_sub_task_completion": [],
            "current_sub_task_code_gen": None,
            "current_sub_task_code_gen_result": None,
            "current_sub_task_code_gen_error_analysis": None,
            "api_event_updated": {},
        }
        state.update(reasoning_state)

        final_state = self.api_location_loop(state)

        # state.update({self.output[0]: final_state["api_calling_strategy"]})
        # state.update({self.output[1]: final_state["api_calling_event"]})
        return state

    def api_location_loop(self, state: dict) -> dict:
        """
        Executes the overall reasoning loop to generate and validate the code.
        First, it will find out the current api calls is whether enough to get the data.
        If not, it will decompose the user input(user demand) into sub-demands, and find out that those api calls' combination calling strategy will satisfy the user demand.

        Workflow 流程：
        循环外部流程：
            1. 用户需求重构
            2. 过滤 api 事件
            3. 分析 api 功能
        反馈循环流程：
            1. 任务planning
            2. 任务 trail and error 循环
                1. 代码生成
                2. 代码执行
                3. 代码结果评估
                4. 代码结果反馈更新 api example
            3. 整体任务评估
        """
        print("API LOCATION LOOP\n\n")

        response = self.user_demand_rephrase(state)
        state["user_rephrased_input"] = response
        print("**USER REPHRASED INPUT**\n\n", state["user_rephrased_input"])

        state = self.api_call_filter(state)
        if len(state["filtered_api_events"]) == 0:
            raise RuntimeError(
                "No relevant API events found that is possible to get the data user wanted"
            )
        print(
            "FIRST ROUND FILTERED_API_EVENTS NUMBER\n\n",
            len(state["filtered_api_events"]),
        )
        # 更具保留的api事件，分析 api 调用策略

        state = self.api_function_analysis(state)
        print("API FUNCTION ANALYSIS\n\n", state["api_function_analysis"])

        while (
            state["outer_loop_iteration"] < self.max_iterations["outer_loop"]
            and not state["satisfied"]
        ):
            # 任务没有满足，还没到最大外部循环限制，则继续外部循环
            state["outer_loop_iteration"] += 1
            print(
                f"--- Outer Outer Reasoning Iteration {state['outer_loop_iteration']} ---"
            )
            print("**PLANNING**\n\n")
            state = self.planning(state)
            if not isinstance(state["task_plan"], dict):
                raise RuntimeError("No sub task found after planning.")
            # return
            # TODO sub-task 在执行过程上是可能存在依赖关系的，所以我们不能简简单单的并行执行任务。目前先这么凑合。在每个 trail 允许任务出现失败，然后在外部planning下一轮循环中，更新补全 sub task 缺失的信息。

            print("TASK PLAN\n\n", state["task_plan"])
            state = self.trail_and_error_loop(state)

            # overall evaluation
            print("**Overall Evaluation**\n\n")
            state = self.overall_evaluation(state)

            if state["satisfied"]:
                break

        return state

    def user_demand_rephrase(self, state: dict) -> dict:
        prompt = PromptTemplate(
            template=USER_DEMAND_REPHRASE,
            # template=USER_DEMAND_REPHRASE_V2,
            input_variables=["user_input", "data_schema"],
        )
        output_parser = StrOutputParser()
        chain = prompt | self.llm_model | output_parser
        response = self.invoke_with_timeout(
            chain,
            {
                "user_input": state["user_prompt"],
                "data_schema": state["schema"],
            },
            self.timeout,
        )
        return response

    def api_call_filter(self, state: dict) -> dict:
        categorized_events = {"fetch/xhr": [], "other": []}
        non_target_domain_events = []
        source_domain = urlparse(state["url"]).netloc

        for event in state["api_events"]:
            url = event.get("url", "")
            domain = urlparse(url).netloc if url else ""
            parsed_event = {
                "domain": domain,
                "url": url,
                "method": event.get("method", ""),
                "request_headers": event.get("request_headers", {}),
                # "response_headers": event.get("response_headers", {}),
                "status": event.get("status"),
                "payload": event.get("payload"),
                "response": event.get("response"),
                "resource_type": event.get("resource_type"),
                # "timestamp": event.get("timestamp", 0),
                # "response_timestamp": event.get("response_timestamp", 0),
            }
            if domain != source_domain:
                non_target_domain_events.append(parsed_event)
            else:
                resource_type = event.get("resource_type", "").lower()
                if resource_type == "fetch" or resource_type == "xhr":
                    categorized_events["fetch/xhr"].append(parsed_event)
                else:
                    categorized_events["other"].append(parsed_event)

        # TODO 基于大模型的api过滤的标准很难制定，需要好好考虑这个问题
        # chain_dict = {}
        # filtered_api_events = []
        # for i, api_event in enumerate(tqdm(categorized_events["fetch/xhr"])):
        #     prompt = PromptTemplate(
        #         template=API_CALL_FILTER,
        #         input_variables=["user_rephrased_input"],
        #         partial_variables={"api_event": api_event},
        #     )
        #     output_parser = BooleanOutputParser()
        #     chain_dict[f"api_filter_{i}"] = prompt | self.llm_model | output_parser
        # async_runner = RunnableParallel(**chain_dict)
        # response = self.invoke_with_timeout(
        #     async_runner,
        #     {
        #         "user_rephrased_input": state["user_rephrased_input"],
        #     },
        #     self.timeout,
        # )
        # print("filter response\n\n", response)
        # for k, v in response.items():
        #     if v:
        #         filtered_api_events.append(state["api_events"][int(k.split("_")[-1])])
        def parse_event(event):
            parsed = {}
            for k, v in event.items():
                if isinstance(v, (dict, list)):
                    parsed[k] = json.dumps(v, ensure_ascii=False)
                elif not isinstance(v, str):
                    parsed[k] = str(v)
                else:
                    parsed[k] = v
            return parsed

        state["filtered_api_events"] = {
            f"filtered_api_{idx}": parse_event(event)
            for idx, event in enumerate(categorized_events["fetch/xhr"])
        }
        print(
            "filtered_api_events\n\n",
            [[k, v["url"]] for k, v in state["filtered_api_events"].items()],
        )
        return state

    def api_function_analysis(self, state: dict) -> dict:
        chain_dict = {}
        for api_key, api_call in state["filtered_api_events"].items():
            prompt = PromptTemplate(
                template=API_FUNCTION_ANALYSIS,
                # template=API_FUNCTION_ANALYSIS_V2,
                input_variables=["api_call"],
                partial_variables={"api_call": api_call},
            )
            output_parser = StrOutputParser()
            chain = prompt | self.llm_model | output_parser
            chain_dict[api_key] = chain
        async_runner = RunnableParallel(**chain_dict)
        response = self.invoke_with_timeout(
            async_runner,
            {},
            self.timeout,
        )
        state["api_function_analysis"] = response
        return state

    def planning(self, state: dict) -> dict:
        """
        根据当前总任务状态和目标，规划下一步任务。
        """

        class SubTask(BaseModel):
            task_name: str = Field(
                description="The name of the sub task, it should be a unique string."
            )
            task_target: str = Field(
                description="The target of the sub task, it should be clear, specific and in details."
            )
            task_details: str = Field(
                description="The detailed planning of the sub task, it should be a detailed description of how to use the api to meet the task target."
            )
            task_api_call_index: str = Field(
                description="The index of the api call in the API FUNCTION ANALYSIS, it should be a string of the key."
            )

        # class TaskPlan(BaseModel):
        #     sub_tasks: List[SubTask]

        output_parser = get_pydantic_output_parser(SubTask)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=PLANNING,
            input_variables=[
                "task_status",
                "user_rephrased_input",
                # "api_call_candidates",
                "api_function_analysis",
                "history_sub_task_completion",
            ],
            partial_variables={"format_instruction": format_instructions},
        )
        chain = prompt | self.llm_model | output_parser
        response = self.invoke_with_timeout(
            chain,
            {
                "task_status": "satisfied" if state["satisfied"] else "unsatisfied",
                "user_rephrased_input": state["user_rephrased_input"],
                # "api_call_candidates": state["filtered_api_events"],
                "api_function_analysis": state["api_function_analysis"],
                "history_sub_task_completion": state["history_sub_task_completion"],
            },
            self.timeout,
        )
        state["task_plan"] = response
        return state

    def trail_and_error_loop(self, state: dict) -> dict:
        """
        给一个明确的目标，写代码，检查代码，执行代码，验证代码，输出结果，检测结果是否满足目标。
        """
        sub_task_trial_max = 6
        task = state["task_plan"]
        # sub task 第一次代码生成
        init_code_gen = self._sub_task_init_code_gen(state, task)
        if (
            len(init_code_gen) == 0
            or init_code_gen == '```python\nprint("No clue on task.")\n```'
        ):
            raise RuntimeError("No valid code generated.")
        state["current_sub_task_code_gen"] = init_code_gen
        print("current_sub_task_code_gen\n\n", state["current_sub_task_code_gen"])

        print("**Code Run**\n\n")
        state["current_sub_task_code_gen_result"] = None
        state["current_sub_task_code_gen_error_analysis"] = None
        trail_count = 0
        while trail_count < sub_task_trial_max:
            code_gen_result = self.code_run_loop(state)
            state["current_sub_task_code_gen_result"] = code_gen_result
            task_eval_result = self.sub_task_output_evaluation(
                state, task, code_gen_result
            )

            if task_eval_result:
                state["history_sub_task_completion"].append(
                    {
                        "task_name": task["task_name"],
                        "task_target": task["task_target"],
                        "task_details": task["task_details"],
                        "task_code": state["current_sub_task_code_gen"],
                        "task_result": code_gen_result,
                        "task_completion_status": task_eval_result,
                    }
                )
                print("SUB TASK EVALUATION SUCCESS\n\n", task["task_name"])
                api_call = state["filtered_api_events"][task["task_api_call_index"]]
                api_call_update = self._update_api_event(
                    api_call,
                    state["current_sub_task_code_gen"],
                    state["current_sub_task_code_gen_result"],
                )
                print("API CALL UPDATE\n\n", api_call_update)
                state["filtered_api_events"][
                    task["task_api_call_index"]
                ] = api_call_update
                break
            else:
                print("SUB TASK EVALUATION FAILED\n\n", task["task_name"])
                sub_task_failed_analysis = self._sub_task_failed_analysis(
                    state,
                    task,
                    state["current_sub_task_code_gen"],
                    state["current_sub_task_code_gen_result"],
                )
                print("SUB TASK FAILED ANALYSIS\n\n", sub_task_failed_analysis)
                state["current_sub_task_code_gen_error_analysis"] = (
                    sub_task_failed_analysis
                )
                re_code_gen = self._sub_task_re_code_gen(state, task)
                state["current_sub_task_code_gen"] = re_code_gen
                trail_count += 1
        return state

    def _sub_task_init_code_gen(self, state: dict, sub_task: dict) -> str:
        api_call = state["filtered_api_events"][sub_task["task_api_call_index"]]
        api_function = state["api_function_analysis"][sub_task["task_api_call_index"]]
        prompt = PromptTemplate(
            template=SUB_TASK_API_CALL_TRAIL,
            input_variables=[
                "user_rephrased_input",
                "task_target",
                "api_call",
                "api_call_function",
            ],
            partial_variables={
                "task_target": sub_task["task_target"]
                + "\n"
                + sub_task["task_details"],
                "api_call": api_call,
                "api_call_function": api_function,
            },
        )
        output_parser = StrOutputParser()

        chain = prompt | self.llm_model | output_parser
        response = self.invoke_with_timeout(
            chain,
            {
                "user_rephrased_input": state["user_rephrased_input"],
            },
            self.timeout,
        )
        return response

    def _sub_task_failed_analysis(
        self,
        state: dict,
        task: dict,
        api_call_code_gen: str,
        api_call_code_execution_result: str,
    ) -> str:
        prompt = PromptTemplate(
            template=SUB_TASK_FAILED_ANALYSIS,
            input_variables=[
                "task_target",
                "api_call_code",
                "api_call_code_execution_result",
                "api_call",
            ],
        )
        chain = prompt | self.llm_model | StrOutputParser()
        return chain.invoke(
            {
                "task_target": task["task_target"] + "\n" + task["task_details"],
                "api_call_code": api_call_code_gen,
                "api_call_code_execution_result": api_call_code_execution_result,
                "api_call": state["filtered_api_events"][task["task_api_call_index"]],
            }
        )

    # def _get_task(self, state: dict, task_name: str) -> dict:
    #     for sub_task in state["task_plan"]["sub_tasks"]:
    #         if sub_task["task_name"] == task_name:
    #             return sub_task
    #     raise RuntimeError(f"Task {task_name} not found.")

    def _sub_task_re_code_gen(self, state: dict, task: dict) -> str:
        prompt = PromptTemplate(
            template=SUB_TASK_API_CALL_TRAIL_REGENERATE,
            input_variables=[
                "task_target",
                "api_call_code_gen",
                "api_call_code_execution_result",
                "api_call_code_error_analysis",
            ],
        )

        chain = prompt | self.llm_model | StrOutputParser()
        response = self.invoke_with_timeout(
            chain,
            {
                "task_target": task["task_target"] + "\n" + task["task_details"],
                "api_call_code_gen": state["current_sub_task_code_gen"],
                "api_call_code_execution_result": state[
                    "current_sub_task_code_gen_result"
                ],
                "api_call_code_error_analysis": state[
                    "current_sub_task_code_gen_error_analysis"
                ],
            },
            self.timeout,
        )
        return response

    def _update_api_event(
        self, api_call, api_call_code, api_call_code_execution_result
    ):
        print(
            "API CALL UPDATE data source\n\n",
            api_call,
            api_call_code,
            api_call_code_execution_result,
        )
        prompt = PromptTemplate(
            template=API_CALL_UPDATE,
            input_variables=[
                "api_call",
                "api_call_code",
                "api_call_code_execution_result",
            ],
        )
        chain = prompt | self.llm_model | StrOutputParser()
        response = self.invoke_with_timeout(
            chain,
            {
                "api_call": api_call,
                "api_call_code": api_call_code,
                "api_call_code_execution_result": api_call_code_execution_result,
            },
            self.timeout,
        )
        return extract_api_call_example(response)

    def sub_task_output_evaluation(
        self, state: dict, task: dict, code_gen_result: str
    ) -> bool:
        prompt = PromptTemplate(
            template=API_OUTPUT_EVALUATION,
            input_variables=["task_target", "api_call_result"],
        )
        chain = prompt | self.llm_model | BooleanOutputParser()
        response = self.invoke_with_timeout(
            chain,
            {
                "task_target": task["task_target"] + "\n" + task["task_details"],
                "api_call_result": code_gen_result,
            },
            self.timeout,
        )
        # print("output_evaluation\n\n", response)
        return response

    def overall_evaluation(self, state: dict) -> bool:

        prompt = PromptTemplate(
            template=OVERALL_EVALUATION,
            input_variables=["user_rephrased_input", "history_sub_task_completion"],
        )
        chain = prompt | self.llm_model | BooleanOutputParser()
        response = self.invoke_with_timeout(
            chain,
            {
                "user_rephrased_input": state["user_rephrased_input"],
                "history_sub_task_completion": state["history_sub_task_completion"],
            },
            self.timeout,
        )
        print("overall_evaluation\n\n", response)
        state["satisfied"] = response
        return state

    def code_run_loop(self, state: dict) -> dict:
        code_gen = state["current_sub_task_code_gen"]
        clean_code = extract_code(code_gen)
        state["generated_code"] = clean_code
        # code syntax check
        for _ in range(self.max_iterations["syntax"]):
            syntax_valid, syntax_message = self.syntax_check(clean_code)
            if syntax_valid:
                state["errors"]["syntax"] = []
                break
            else:
                state["errors"]["syntax"] = [syntax_message]
                print("SYNTAX ERROR\n", state["errors"]["syntax"])
                analysis = syntax_focused_analysis(state, self.llm_model)
                print("SYNTAX ANALYSIS\n", analysis)
                state["generated_code"] = syntax_focused_code_generation(
                    state, analysis, self.llm_model
                )
        # code execution check
        for _ in range(self.max_iterations["execution"]):
            execution_success, execution_result = self.create_sandbox_and_execute(
                state["generated_code"]
            )
            if execution_success:
                state["errors"]["execution"] = []
                print("EXECUTION SUCCESS, RESULT\n\n", execution_result)
                return execution_result
            else:
                state["errors"]["execution"] = [execution_result]
                print("EXECUTION ERROR\n", state["errors"]["execution"])
                analysis = api_execution_focused_analysis(state, self.llm_model)
                print("EXECUTION ANALYSIS\n", analysis)
                state["generated_code"] = execution_focused_code_generation(
                    state, analysis, self.llm_model
                )
        if state["errors"]["execution"] != []:
            raise RuntimeError("Execution error is not resolved.")

    def syntax_check(self, code):
        """
        Checks the syntax of the provided code.

        Args:
            code (str): The code to be checked for syntax errors.

        Returns:
            tuple: A tuple containing a boolean indicating if the syntax is correct and a message.
        """
        try:
            ast.parse(code)
            return True, "Syntax is correct."
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"

    def create_sandbox_and_execute(self, function_code):
        """
        创建沙箱环境并执行提供的函数代码。

        参数:
            function_code (str): 要在沙箱中执行的代码。

        返回:
            tuple: 包含执行是否成功的布尔值，以及结果或错误消息的元组。
        """
        sandbox_globals = {
            "json": json,
            "re": re,
            "time": time,
            "requests": requests,
            "__builtins__": __builtins__,
        }

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            exec(function_code, sandbox_globals)

            extract_data = sandbox_globals.get("api_call_test")

            if not extract_data:
                raise NameError("在生成的代码中未找到'api_call_test'函数。")

            result = extract_data()
            return True, result
        except Exception as e:
            import traceback

            error_traceback = traceback.format_exc()
            return (
                False,
                f"Execution error: {str(e)}\n\n Error Traceback:\n{error_traceback}",
            )
        finally:
            sys.stdout = old_stdout


def api_execution_focused_analysis(state: dict, llm_model) -> str:
    prompt = PromptTemplate(
        template=API_EXECUTION_ERROR_ANALYSIS,
        input_variables=[
            "generated_code",
            "errors",
        ],
    )
    chain = prompt | llm_model | StrOutputParser()
    return chain.invoke(
        {
            "generated_code": state["generated_code"],
            "errors": state["errors"]["execution"],
        }
    )
