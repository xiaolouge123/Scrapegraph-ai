from typing import Optional
from pydantic import BaseModel
from .base_graph import BaseGraph
from .abstract_graph import AbstractGraph
from ..utils.save_code_to_file import save_code_to_file

from ..nodes import (
    FetchAPINode,
    ParseAPINode,
    GenerateAnswerAPINode,
    GenerateCodeAPINode,
    ApiAnalyzerNode,
)


class CodeGeneratorAPIGraph(AbstractGraph):
    """
    CodeGeneratorAPIGraph is a script generator pipeline that generates
    the function extract_data(api_events: list[dict]) -> dict() for
    extracting the wanted information from a list of API events.
    It requires a user prompt, a source URL, and an output schema.

    Attributes:
        prompt (str): The prompt for the graph.
        source (str): The source of the graph.
        config (dict): Configuration parameters for the graph.
        schema (BaseModel): The schema for the graph output.
        llm_model: An instance of a language model client, configured for generating answers.
        verbose (bool): A flag indicating whether to show print statements during execution.
        headless (bool): A flag indicating whether to run the graph in headless mode.

    Args:
        prompt (str): The prompt for the graph.
        source (str): The source of the graph.
        config (dict): Configuration parameters for the graph.
        schema (BaseModel): The schema for the graph output.

    Example:
        >>> code_gen = CodeGeneratorAPIGraph(
        ...     "Find all the data loaded from the API",
        ...     "https://en.wikipedia.org/wiki/Chioggia",
        ...     {"llm": {"model": "openai/gpt-3.5-turbo"}}
        ... )
        >>> result = code_gen.run()
    """

    def __init__(
        self, prompt: str, source: str, config: dict, schema: Optional[BaseModel] = None
    ):
        super().__init__(prompt, config, source, schema)
        assert (
            self.input_key == "url"
        ), "The input key must be 'url' for CodeGeneratorAPIGraph"

    def _create_graph(self) -> BaseGraph:
        """
        Creates the graph of nodes representing the workflow for api calls scraping.

        Returns:
            BaseGraph: A graph instance representing the api calls scraping workflow.
        """
        if self.schema is None:
            raise KeyError("The schema is required for CodeGeneratorAPIGraph")

        fetch_api_node = FetchAPINode(
            input="url",
            output=["api_events"],
            node_config=None,
        )

        parse_api_node = ParseAPINode(
            input="api_events",
            output=["parsed_api_events"],
            node_config=None,
        )

        generate_answer_api_node = GenerateAnswerAPINode(
            input="user_prompt & parsed_api_events",
            output=["answer"],
            node_config=None,
        )

        api_analyzer_node = ApiAnalyzerNode(
            input="answer",
            output=["api_analysis"],
            node_config=None,
        )

        generate_code_api_node = GenerateCodeAPINode(
            input="user_prompt & answer",
            output=["code"],
            node_config=None,
        )

        return BaseGraph(
            nodes=[
                fetch_api_node,
                parse_api_node,
                generate_answer_api_node,
                api_analyzer_node,
                generate_code_api_node,
            ],
            edges=[
                (fetch_api_node, parse_api_node),
                (parse_api_node, generate_answer_api_node),
                (generate_answer_api_node, api_analyzer_node),
                (api_analyzer_node, generate_code_api_node),
            ],
            entry_point=fetch_api_node,
            graph_name=self.__class__.__name__,
        )

    def run(self) -> str:
        """
        Executes the api scraping process and returns the generated code.

        Returns:
            str: The generated code.
        """
        inputs = {"user_prompt": self.prompt, self.input_key: self.source}
        self.final_state, self.execution_info = self.graph.execute(inputs)

        generated_code = self.final_state.get("code", "No code created.")

        if self.config.get("filename") is None:
            filename = "extracted_data.py"
        elif ".py" not in self.config.get("filename"):
            filename += ".py"
        else:
            filename = self.config.get("filename")

        save_code_to_file(generated_code, filename)

        return generated_code
