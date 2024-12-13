"""
FetchAPINode Module
"""

from .base_node import BaseNode
from ..utils.logging import get_logger
from ..docloaders import ChromiumLoader
from typing import List, Optional, Dict, Any
import json
import asyncio
from playwright.async_api import async_playwright, Request, Response


async def capture_api_events(url: str, headless: bool = True) -> tuple[str, List[Dict]]:
    """
    使用 Playwright 捕获页面加载过程中的 API 请求事件。

    Args:
        url: 要访问的网页 URL
        headless: 是否使用无头模式

    Returns:
        tuple: (页面内容, API 事件列表)
    """
    # 使用字典存储请求-响应对
    request_map = {}
    api_events = []

    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=headless)
        page = await browser.new_page()

        async def handle_request(request: Request):
            request_data = {
                "url": request.url,
                "method": request.method,
                "request_headers": request.headers,
                "payload": request.post_data,
                "timestamp": request.timing.get("startTime", 0),
                "resource_type": request.resource_type,
            }
            request_map[request.url] = request_data

        async def handle_response(response: Response):
            url = response.url
            if url in request_map:
                # 合并请求和响应数据
                try:
                    # 尝试获取 JSON 响应
                    response_data = await response.json()
                except:
                    try:
                        # 如果 JSON 解析失败，获取原始响应并尝试处理编码
                        response_body = await response.body()
                        content_type = response.headers.get("content-type", "")
                        print(content_type)

                        # 从 content-type 中提取字符集
                        charset = "utf-8"  # 默认编码
                        if "charset=" in content_type.lower():
                            charset = (
                                content_type.lower().split("charset=")[-1].split(";")[0]
                            )

                        # 使用检测到的编码解码响应内容
                        response_data = response_body.decode(charset, errors="replace")

                        # 如果响应看起来像 JSON，尝试解析它
                        if response_data.strip().startswith(
                            "{"
                        ) or response_data.strip().startswith("["):
                            response_data = json.loads(response_data)
                    except Exception as e:
                        # 如果所有尝试都失败，存储错误信息
                        response_data = f"无法解析响应内容: {str(e)}"

                event_data = request_map[url]
                event_data.update(
                    {
                        "status": response.status,
                        "response": response_data,
                        "response_headers": response.headers,
                        "response_timestamp": response.request.timing.get(
                            "responseEnd", 0
                        ),
                    }
                )
                api_events.append(event_data)
                # 清理已处理的请求
                del request_map[url]

        # 监听请求和响应事件
        page.on("request", handle_request)
        page.on("response", handle_response)

        # 访问页面
        await page.goto(url, wait_until="networkidle")
        content = await page.content()

        await browser.close()

        return content, api_events


class FetchAPINode(BaseNode):
    """
    A specialized FetchNode that monitors and logs API calls during URL fetching and HTML loading.
    Provides detailed information about network requests, responses, and page load events.

    Attributes:
        api_events (List[Dict]): List of API events captured during page load
        network_stats (Dict): Statistics about network performance
        request_headers (Dict): Custom headers for requests
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "FetchAPI",
    ):
        super().__init__(node_name, "node", input, output, 1, node_config)
        self.headless = (
            True if node_config is None else node_config.get("headless", True)
        )
        self.verbose = (
            False if node_config is None else node_config.get("verbose", False)
        )

        self.logger = get_logger(__name__)

    def is_valid_url(self, source: str) -> bool:
        """
        Validates if the source string is a valid URL using regex.

        Parameters:
        source (str): The URL string to validate

        Raises:
        ValueError: If the URL is invalid
        """
        import re

        url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        if not bool(re.match(url_pattern, source)):
            raise ValueError(
                f"Invalid URL format: {source}. URL must start with http(s):// and contain a valid domain."
            )
        return True

    def execute(self, state):
        """
        Executes the node's logic to fetch api calls from a specified URL and
        update the state with this content.
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        input_keys = self.get_input_keys(state)
        input_data = [state[key] for key in input_keys]
        source = input_data[0]
        input_type = input_keys[0]

        try:
            if self.is_valid_url(source):
                return self.handle_web_source(state, source)
        except ValueError as e:
            # Re-raise the exception from is_valid_url
            raise
        return state

    def handle_web_source(self, state, source):
        """
        Enhanced web source handler with API monitoring.
        """
        self.api_events = []  # Reset events for new request
        self.network_stats = {
            "total_requests": 0,
            "failed_requests": 0,
            "total_bytes": 0,
        }
        loader_kwargs = {}

        if self.node_config:
            loader_kwargs = self.node_config.get("loader_kwargs", {})

        try:
            content, api_events = asyncio.run(capture_api_events(source, self.headless))
            state.update({"content": content})
            state.update({self.output[0]: api_events})
            return state

        except Exception as e:
            self.logger.error(f"Error fetching URL {source}: {str(e)}")
            raise ValueError(f"Error fetching URL {source}: {str(e)}")
        return state
