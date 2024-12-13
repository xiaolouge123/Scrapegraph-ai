from typing import Optional, List, Any, Dict
from .base_node import BaseNode
from ..utils import (
    transform_schema,
    extract_code,
    syntax_focused_analysis,
    syntax_focused_code_generation,
    execution_focused_code_generation,
    validation_focused_analysis,
    validation_focused_code_generation,
    semantic_focused_analysis,
    semantic_focused_code_generation,
    are_content_equal,
)
from ..utils.code_error_analysis import api_execution_focused_analysis
from ..utils.output_parser import get_pydantic_output_parser
from ..prompts.generate_code_api_node_prompts import (
    TEMPLATE_INIT_CODE_API_GENERATION,
    API_REFERENCE_DATA_GENERATION,
)
from ..prompts import TEMPLATE_SEMANTIC_COMPARISON
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import json
import re
import time
import sys
from io import StringIO
import requests
from jsonschema import validate, ValidationError
import ast
import logging


class GenerateCodeAPINode(BaseNode):
    """
    A node that generates Python code for a function that extracts data
    from api calls.

    Attributes:
        llm_model: An instance of a language model client, configured for generating answers.
        verbose (bool): A flag indicating whether to show print statements during execution.
    Args:
        input (str): Boolean expression defining the input keys needed from the state.
        output (List[str]): List of output keys to be updated in the state.
        node_config (dict): Additional configuration for the node.
        node_name (str): The unique identifier name for the node, defaulting to "GenerateCodeAPI".
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "GenerateCodeAPI",
    ):
        super().__init__(node_name, "node", input, output, 4, node_config)

        self.llm_model = node_config["llm_model"]
        self.verbose = (
            True if node_config is None else node_config.get("verbose", False)
        )

        self.max_iterations = node_config.get(
            "max_iterations",
            {
                "overall": 10,
                "syntax": 3,
                "execution": 3,
                "validation": 3,
                "semantic": 3,
            },
        )
        self.output_schema = node_config.get("schema")

    def execute(self, state: dict) -> dict:
        """
        Generates Python code for a function that extracts data from api calls.

        Args:
            state (dict): The current state of the graph. The input keys will be used to fetch the correct data from the state.

        Returns:
            dict: The updated state with the output key containing the generated answer.

        Raises:
            KeyError: If the input keys are not found in the state, indicating that the necessary information for generating an answer is missing.
            RuntimeError: If the maximum number of iterations is reached without obtaining the desired code.
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        input_data = [state[key] for key in input_keys]

        url = input_data[0]
        user_prompt = input_data[1]
        api_analysis = input_data[2]
        candidates_api_events = input_data[3]

        simplefied_schema = str(transform_schema(self.output_schema.schema()))

        reasoning_state = {
            "url": url,
            "user_input": user_prompt,
            "json_schema": simplefied_schema,
            "api_analysis": api_analysis,
            "candidates_api_events": candidates_api_events,
            "errors": {"syntax": [], "execution": [], "validation": [], "semantic": []},
            "iteration": 0,
        }

        final_state = self.overall_reasoning_loop(reasoning_state)

        state.update({self.output[0]: final_state["generated_code"]})
        return state

    def overall_reasoning_loop(self, state: dict) -> dict:
        """
        Executes the overall reasoning loop to generate and validate the code.

        Args:
            state (dict): The current state of the reasoning process.

        Returns:
            dict: The final state after the reasoning loop.

        Raises:
            RuntimeError: If the maximum number of iterations is reached without obtaining the desired code.
        """
        self.logger.info(f"--- (Generating Code API) ---")
        state["generated_code"] = self.generate_initial_code(state)
        state["generated_code"] = extract_code(state["generated_code"])
        print("INIT CODE\n", state["generated_code"])
        state["reference_answer"] = self.pre_generate_reference_data(state)

        while state["iteration"] < self.max_iterations["overall"]:
            state["iteration"] += 1
            if self.verbose:
                self.logger.info(f"--- Iteration {state['iteration']} ---")

            self.logger.info(f"--- (Checking Code Syntax) ---")
            print("SYNTAX REASONING LOOP\n\n")
            state = self.syntax_reasoning_loop(state)
            if state["errors"]["syntax"]:
                continue

            self.logger.info(f"--- (Executing the Generated Code) ---")
            print("EXECUTION REASONING LOOP\n\n")
            state = self.execution_reasoning_loop(state)
            if state["errors"]["execution"]:
                continue

            self.logger.info(f"--- (Validate the Code Output Schema) ---")
            print("VALIDATION REASONING LOOP\n\n")
            state = self.validation_reasoning_loop(state)
            if state["errors"]["validation"]:
                continue

            self.logger.info(
                f"""--- (Checking if the informations
                             exctrcated are the ones Requested) ---"""
            )
            state = self.semantic_comparison_loop(state)
            if state["errors"]["semantic"]:
                continue
            break

        if state["iteration"] == self.max_iterations["overall"] and (
            state["errors"]["syntax"]
            or state["errors"]["execution"]
            or state["errors"]["validation"]
            or state["errors"]["semantic"]
        ):
            raise RuntimeError(
                "Max iterations reached without obtaining the desired code."
            )

        self.logger.info(f"--- (Code API Generated Correctly) ---")

        return state

    def pre_generate_reference_data(self, state: dict) -> dict:
        """
        Generate the reference data for semantic comparison.
        Based on the given information like: user prompt, json schema, and api events. Try to generate the reference data or at least the partial reference data.

        Args:
            state (dict): The current state of the reasoning process.

        Returns:
            dict: The updated state with the reference data.
        """
        prompt = PromptTemplate(
            template=API_REFERENCE_DATA_GENERATION,
            input_variables=[
                "user_input",
                "json_schema",
                "candidates_api_events",
            ],
        )
        if self.node_config.get("schema", None) is not None:
            output_parser = get_pydantic_output_parser(self.node_config["schema"])
            format_instructions = output_parser.get_format_instructions()
        else:
            output_parser = JsonOutputParser()
            format_instructions = output_parser.get_format_instructions()

        chain = prompt | self.llm_model | output_parser
        reference_answer = chain.invoke(
            {
                "user_input": state["user_input"],
                "json_schema": format_instructions,
                "candidates_api_events": state["candidates_api_events"],
            }
        )
        return reference_answer

    def generate_initial_code(self, state: dict) -> str:
        """
        Generates the initial code based on the provided state.

        Args:
            state (dict): The current state of the reasoning process.

        Returns:
            str: The initially generated code.
        """
        prompt = PromptTemplate(
            template=TEMPLATE_INIT_CODE_API_GENERATION,
            partial_variables={
                "user_input": state["user_input"],
                "json_schema": state["json_schema"],
                "api_analysis": state["api_analysis"],
                "candidates_api_events": state["candidates_api_events"],
            },
        )
        output_parser = StrOutputParser()

        chain = prompt | self.llm_model | output_parser
        generated_code = chain.invoke({})
        return generated_code

    def syntax_reasoning_loop(self, state: dict) -> dict:
        """
        Executes the syntax reasoning loop to ensure the generated code has correct syntax.

        Args:
            state (dict): The current state of the reasoning process.

        Returns:
            dict: The updated state after the syntax reasoning loop.
        """
        for _ in range(self.max_iterations["syntax"]):
            syntax_valid, syntax_message = self.syntax_check(state["generated_code"])
            if syntax_valid:
                state["errors"]["syntax"] = []
                return state

            state["errors"]["syntax"] = [syntax_message]
            print("SYNTAX ERROR\n", state["errors"]["syntax"])
            self.logger.info(f"--- (Synax Error Found: {syntax_message}) ---")
            analysis = syntax_focused_analysis(state, self.llm_model)
            print("SYNTAX ANALYSIS\n", analysis)
            self.logger.info(
                f"""--- (Regenerating Code
                             to fix the Error) ---"""
            )
            state["generated_code"] = syntax_focused_code_generation(
                state, analysis, self.llm_model
            )
            state["generated_code"] = extract_code(state["generated_code"])
            print("SYNTAX REGENERATED CODE\n", state["generated_code"])
        return state

    def execution_reasoning_loop(self, state: dict) -> dict:
        """
        Executes the execution reasoning loop to ensure the generated code runs without errors.

        Args:
            state (dict): The current state of the reasoning process.

        Returns:
            dict: The updated state after the execution reasoning loop.
        """
        for _ in range(self.max_iterations["execution"]):
            execution_success, execution_result = self.create_sandbox_and_execute(
                state["generated_code"]
            )
            if execution_success:
                state["execution_result"] = execution_result
                print("EXECUTION SUCCESS, RESULT\n\n", state["execution_result"])
                state["errors"]["execution"] = []
                return state

            state["errors"]["execution"] = [execution_result]
            print("EXECUTION ERROR\n\n", state["errors"]["execution"])
            self.logger.info(f"--- (Code Execution Error: {execution_result}) ---")
            analysis = api_execution_focused_analysis(state, self.llm_model)
            print("EXECUTION ANALYSIS\n\n", analysis)
            self.logger.info(f"--- (Regenerating Code to fix the Error) ---")
            state["generated_code"] = execution_focused_code_generation(
                state, analysis, self.llm_model
            )
            state["generated_code"] = extract_code(state["generated_code"])
            print("EXECUTION REGENERATED CODE\n\n", state["generated_code"])
        return state

    def validation_reasoning_loop(self, state: dict) -> dict:
        """
        Executes the validation reasoning loop to ensure the
        generated code's output matches the desired schema.

        Args:
            state (dict): The current state of the reasoning process.

        Returns:
            dict: The updated state after the validation reasoning loop.
        """
        for _ in range(self.max_iterations["validation"]):
            validation, errors = self.validate_dict(
                state["execution_result"], self.output_schema.schema()
            )
            if validation:
                state["errors"]["validation"] = []
                return state

            state["errors"]["validation"] = errors
            print("VALIDATION ERROR\n\n", state["errors"]["validation"])
            self.logger.info(
                f"--- (Code Output not compliant to the deisred Output Schema) ---"
            )
            analysis = validation_focused_analysis(state, self.llm_model)
            print("VALIDATION ANALYSIS\n\n", analysis)
            self.logger.info(
                f"""--- (Regenerating Code to make the
                             Output compliant to the deisred Output Schema) ---"""
            )
            state["generated_code"] = validation_focused_code_generation(
                state, analysis, self.llm_model
            )
            state["generated_code"] = extract_code(state["generated_code"])
            print("VALIDATION REGENERATED CODE\n\n", state["generated_code"])
        return state

    def semantic_comparison_loop(self, state: dict) -> dict:
        """
        Executes the semantic comparison loop to ensure the generated code's
          output is semantically equivalent to the reference answer.

        Args:
            state (dict): The current state of the reasoning process.

        Returns:
            dict: The updated state after the semantic comparison loop.
        """
        for _ in range(self.max_iterations["semantic"]):
            comparison_result = self.semantic_comparison(
                state["execution_result"], state["reference_answer"]
            )
            if comparison_result["are_semantically_equivalent"]:
                state["errors"]["semantic"] = []
                return state

            state["errors"]["semantic"] = comparison_result["differences"]
            print("SEMANTIC ERROR\n", state["errors"]["semantic"])
            self.logger.info(
                f"""--- (The informations exctrcated
                             are not the all ones requested) ---"""
            )
            analysis = semantic_focused_analysis(
                state, comparison_result, self.llm_model
            )
            print("SEMANTIC ANALYSIS\n", analysis)
            self.logger.info(
                f"""--- (Regenerating Code to
                                obtain all the infromation requested) ---"""
            )
            state["generated_code"] = semantic_focused_code_generation(
                state, analysis, self.llm_model
            )
            state["generated_code"] = extract_code(state["generated_code"])
            print("SEMANTIC REGENERATED CODE\n", state["generated_code"])
        return state

    def semantic_comparison(
        self, generated_result: Any, reference_result: Any
    ) -> Dict[str, Any]:
        """
        Performs a semantic comparison between the generated result and the reference result.

        Args:
            generated_result (Any): The result generated by the code.
            reference_result (Any): The reference result for comparison.

        Returns:
            Dict[str, Any]: A dictionary containing the comparison result,
            differences, and explanation.
        """
        reference_result_dict = self.output_schema(**reference_result).dict()
        print("GENERATED RESULT\n", generated_result)
        print("REFERENCE ANSWER\n", reference_result_dict)
        if are_content_equal(generated_result, reference_result_dict):
            return {
                "are_semantically_equivalent": True,
                "differences": [],
                "explanation": "The generated result and reference result are exactly equal.",
            }

        response_schemas = [
            ResponseSchema(
                name="are_semantically_equivalent",
                description="""Boolean indicating if the
                           results are semantically equivalent""",
            ),
            ResponseSchema(
                name="differences",
                description="""List of semantic differences
                           between the results, if any""",
            ),
            ResponseSchema(
                name="explanation",
                description="""Detailed explanation of the
                           comparison and reasoning""",
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        prompt = PromptTemplate(
            template=TEMPLATE_SEMANTIC_COMPARISON,
            input_variables=["generated_result", "reference_result"],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            },
        )

        chain = prompt | self.llm_model | output_parser
        return chain.invoke(
            {
                "generated_result": json.dumps(generated_result, indent=2),
                "reference_result": json.dumps(reference_result_dict, indent=2),
            }
        )

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

            extract_data = sandbox_globals.get("extract_data")

            if not extract_data:
                raise NameError("在生成的代码中未找到'extract_data'函数。")

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

    def validate_dict(self, data: dict, schema):
        """
        Validates the provided data against the given schema.

        Args:
            data (dict): The data to be validated.
            schema (dict): The schema against which the data is validated.

        Returns:
            tuple: A tuple containing a boolean indicating
            if the validation was successful and a list of errors if any.
        """
        try:
            validate(instance=data, schema=schema)
            return True, None
        except ValidationError as e:
            errors = [e.message]
            return False, errors
