from typing import Dict, List, Any
from .base_node import BaseNode
import time
from tqdm import tqdm
from requests.exceptions import Timeout
from urllib.parse import urlparse
from ..prompts.api_analyzer_node_prompts import (
    API_RELEVANT_ANALYSIS,
    API_GENERAL_FUNCTION,
    API_REFERENCE_ANSWER,
    TEMPLATE_API_ANALYSIS_MERGE_API_CALLS,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import BooleanOutputParser
from ..utils.output_parser import get_pydantic_output_parser


class APIAnalyzerNode(BaseNode):
    """
    A node responsible for parsing API events from raw fetch api results.
    Categorizes API requests by type and domain, with special focus on fetch and XHR requests.
    This Node will first categorize the API events on the domain level, and then on the request type level.
    The request domain that is not the same as the user input url's domain will be categorized as "other" and only take consideration when the normal analysis is at a dead end.
    The request type level will categorize the API events into fetch/xhr and other request types will be categorized as "other". Fetch and xhr request will be focused on the most to diagnose which requests satisfy the user demands on target date.
    The other requests will be taken consideration when the normal analysis is at a dead end.

    Args:

    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Dict | None = None,
        node_name: str = "APIAnalyzerNode",
    ):
        super().__init__(node_name, "node", input, output, 3, node_config)

        self.llm_model = node_config["llm_model"]
        self.verbose = node_config.get("verbose", False)
        self.timeout = node_config.get("timeout", 120)

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
        source_domain = urlparse(url).netloc if url else ""
        user_prompt = input_data[1]
        # self.logger.info(f"input data: {input_data}")
        # print("input data: ", len(input_data))
        # print("input data 0: ", input_data[0])
        # print("input data 1: ", input_data[1])
        # print("input data 2: ", type(input_data[2]))
        api_events = input_data[2]

        if self.node_config.get("schema", None) is not None:
            model_json_schema = self.node_config["schema"].model_json_schema()
            output_parser = get_pydantic_output_parser(self.node_config["schema"])
            format_instructions = output_parser.get_format_instructions()

        # 构造混合 output parser
        str_output_parser = StrOutputParser()
        bool_output_parser = BooleanOutputParser()

        categorized_events = {"fetch/xhr": [], "other": []}
        non_target_domain_events = []

        # 基础API归纳
        for event in api_events:
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
                "timestamp": event.get("timestamp", 0),
                "response_timestamp": event.get("response_timestamp", 0),
            }
            if domain != source_domain:
                non_target_domain_events.append(parsed_event)
            else:
                resource_type = event.get("resource_type", "").lower()
                if resource_type == "fetch" or resource_type == "xhr":
                    categorized_events["fetch/xhr"].append(parsed_event)
                else:
                    categorized_events["other"].append(parsed_event)

        self.logger.info(
            f"categorized_events stats: fetch/xhr: {len(categorized_events['fetch/xhr'])}, other: {len(categorized_events['other'])}"
        )
        self.logger.info(
            f"non_target_domain_events stats: {len(non_target_domain_events)}"
        )
        # 直接汇总分析每个 api 的 function
        chains_dict = {}
        if len(categorized_events["fetch/xhr"]) > 0:
            for i, event in enumerate(
                tqdm(
                    categorized_events["fetch/xhr"],
                    desc="Processing API events",
                    disable=not self.verbose,
                )
            ):
                event = self._format_event(event, min_model=True)
                prompt = PromptTemplate(
                    template=API_GENERAL_FUNCTION,
                    input_variables=["api_event"],
                    partial_variables={"api_event": event},
                )
                chains_dict[f"api_analysis_{i}"] = (
                    prompt | self.llm_model | str_output_parser
                )

            async_runner = RunnableParallel(**chains_dict)
            try:
                batch_results = self.invoke_with_timeout(async_runner, {}, self.timeout)
            except Timeout as e:
                self.logger.error(f"Timeout error: {str(e)}")
                raise
            # print(batch_results)
            # exit()
            api_analysis_results = {}
            api_candidate_events = {}
            for k, v in batch_results.items():
                index = int(k.split("_")[-1])
                api_analysis_results[k] = v
                api_candidate_events[k] = categorized_events["fetch/xhr"][index]

            state.update({self.output[0]: api_analysis_results})
            state.update({self.output[1]: api_candidate_events})

        return state

        # # 使用 LLM 对 API call 进行筛选
        # # 相关API分析
        # chains_dict = {}
        # relevant_api_index = []

        # if len(categorized_events["fetch/xhr"]) > 0:
        #     for i, event in enumerate(
        #         tqdm(
        #             categorized_events["fetch/xhr"],
        #             desc="Processing API events",
        #             disable=not self.verbose,
        #         )
        #     ):
        #         event = self._format_event(event, min_model=True)
        #         prompt = PromptTemplate(
        #             template=API_RELEVANT_ANALYSIS,
        #             input_variables=["user_prompt", "schema"],
        #             partial_variables={"api_event": event},
        #         )
        #         chain_name = f"api_analysis_{i}"
        #         chains_dict[chain_name] = prompt | self.llm_model | bool_output_parser

        #     async_runner = RunnableParallel(**chains_dict)
        #     try:
        #         relevent_batch_results = self.invoke_with_timeout(
        #             async_runner,
        #             {
        #                 "user_prompt": user_prompt,
        #                 "schema": model_json_schema,
        #             },
        #             self.timeout,
        #         )
        #     except Timeout as e:
        #         self.logger.error(f"Timeout error: {str(e)}")
        #         raise
        #     print(relevent_batch_results)

        #     # API 功能分析
        #     chains_dict = {}
        #     for k, v in relevent_batch_results.items():
        #         index = int(k.split("_")[-1])
        #         if v:
        #             relevant_api_index.append(index)
        #             event = self._format_event(
        #                 categorized_events["fetch/xhr"][index], min_model=True
        #             )
        #             prompt = PromptTemplate(
        #                 template=API_GENERAL_FUNCTION,
        #                 input_variables=["api_event"],
        #                 partial_variables={
        #                     "api_event": categorized_events["fetch/xhr"][index]
        #                 },
        #             )
        #             chains_dict[k] = prompt | self.llm_model | str_output_parser

        #     async_runner = RunnableParallel(**chains_dict)
        #     try:
        #         function_batch_results = self.invoke_with_timeout(
        #             async_runner,
        #             {},
        #             self.timeout,
        #         )
        #     except Timeout as e:
        #         self.logger.error(f"Timeout error: {str(e)}")
        #         raise
        #     print(function_batch_results)
        # exit()
        # # 直接更具 api response 尝试提取参考结果 reference result
        # # TODO: 如果单个请求可以满足用户取数需求，那最好了这里直接挑出这个请求就好。如果用户的取数请求需要从多个 api 接口中取数，组装，再请求，则需要设计一个多层次推理的过程。这里这个抽取参考答案的地方也还有点问题，如果当前 api 可以获得全部数据，也许还行，如果不能获取全部数据，或者获取不到完整 schema 定义字段怎么办，需要请求多个接口拼接后才能获得全部结果。
        # # 设定一个场景问题，用户的数据需求可以通过api请求满足，1. 单个请求即可以获取全部数据和全部字段。 2. 单个请求可以获取全部字段，但需要调整请求参数进行多次请求获取全部数据。3. 需要多个请求协同获取数据，有接口顺序请求依赖，需要顺序的在不同接口请求返回中获取中间结果，用于最终结果的请求。 4. 需要多个请求协同获取数据，不同接口获取不同数据字段，最终汇总后才能拼接完整全部数据字段。 5. 以上各种 case 的有机组合导致的复杂情况。
        # # reference result generation
        # chains_dict = {}
        # for k, v in relevent_batch_results.items():
        #     index = int(k.split("_")[-1])
        #     if v:
        #         event = self._format_event(
        #             categorized_events["fetch/xhr"][index], min_model=True
        #         )
        #         prompt = PromptTemplate(
        #             template=API_REFERENCE_ANSWER,
        #             input_variables=["user_prompt", "format_instructions"],
        #             partial_variables={"api_event": event},
        #         )
        #         chains_dict[k] = prompt | self.llm_model | output_parser

        # async_runner = RunnableParallel(**chains_dict)
        # try:
        #     reference_batch_results = self.invoke_with_timeout(
        #         async_runner,
        #         {
        #             "user_prompt": user_prompt,
        #             "format_instructions": format_instructions,
        #         },
        #         self.timeout,
        #     )
        # except Timeout as e:
        #     self.logger.error(f"Timeout error: {str(e)}")
        #     raise
        # print(reference_batch_results)
        # exit()
        #     relevant_api_calls = []
        #     non_relevant_api_calls = []
        #     for k, v in batch_results.items():
        #         if v:
        #             # relevant api call
        #             i = int(k.split("_")[-1])
        #             relevant_api_calls.append(
        #                 {"api_event": categorized_events["fetch/xhr"][i]}
        #             )
        #         else:
        #             # non relevant api call
        #             i = int(k.split("_")[-1])
        #             non_relevant_api_calls.append(
        #                 {"api_event": categorized_events["fetch/xhr"][i]}
        #             )
        #     merge_api_analysis_prompt = PromptTemplate(
        #         template=TEMPLATE_API_ANALYSIS_MERGE_API_CALLS,
        #         input_variables=["user_prompt", "candidates_api_events", "schema"],
        #     )
        #     merge_chain = merge_api_analysis_prompt | self.llm_model | StrOutputParser()

        #     try:
        #         answer = self.invoke_with_timeout(
        #             merge_chain,
        #             {
        #                 "user_prompt": user_prompt,
        #                 "candidates_api_events": relevant_api_calls,
        #                 "schema": model_json_schema,
        #             },
        #             self.timeout,
        #         )
        #     except Timeout:
        #         state.update(
        #             {
        #                 self.output[0]: {
        #                     "error": "Response timeout exceeded during merge"
        #                 }
        #             }
        #         )
        #         return state
        #     # ["api_analysis", "candidate_api_events"],
        #     state.update({self.output[0]: answer})
        #     state.update({self.output[1]: relevant_api_calls})

        # return state

    def _format_event(self, event: dict, min_model: bool = False) -> dict:
        """Format the event to be used in the prompt
        just now a fake one.
        """
        if min_model:
            _event = {
                "url": event.get("url", ""),
                "method": event.get("method", ""),
                "payload": event.get("payload", {}),
                # "status": event.get("status", ""),
                "response": event.get("response", {}),
            }
            return _event
        else:
            return event
