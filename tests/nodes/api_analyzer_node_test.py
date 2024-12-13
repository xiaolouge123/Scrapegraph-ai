from scrapegraphai.nodes import APIAnalyzerNode, FetchAPINode
import os
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List


class RegionDetail(BaseModel):
    year: int = Field(description="统计年份")
    district_count: int = Field(description="地级区划数")
    city_count: int = Field(description="地级市数")
    county_district_count: int = Field(description="县级区划数")
    municipal_district_count: int = Field(description="市辖区数")
    county_level_city_count: int = Field(description="县级市数")
    county_count: int = Field(description="县数")
    autonomous_county_count: int = Field(description="自治县数")


class RegionDetailCollection(BaseModel):
    stats: List[RegionDetail] = Field(description="行政区划统计数据列表")


class CarInfo(BaseModel):
    name: str = Field(description="车型名称")
    sales: int = Field(description="销量")
    price_range: str = Field(description="价格区间")
    factory: str = Field(description="厂商")


class CarRanking(BaseModel):
    cars: List[CarInfo] = Field(description="车型信息列表")


def test_api_analyzer_node():
    url = "https://data.stats.gov.cn/easyquery.htm?cn=C01"
    user_prompt = "请根据当前页面, 提取近 5 年符合 schema 定义的数据"
    # url = "https://www.dongchedi.com/sales"
    # user_prompt = "请根据当前页面, 提取销量排行榜中第 5 到第 9 位的车型信息，输出需要符合schema 信息"
    state = {
        "url": url,
        "user_prompt": user_prompt,
    }

    node_config = {
        "verbose": True,
        "headless": False,
    }
    openai_key = os.getenv("OPENAI_API_KEY")
    llm_config = {
        "api_key": openai_key,
        "model_provider": "openai",
        "model": "gpt-4o-mini",
        "base_url": "http://localhost:23323/v1",
    }
    llm_model = init_chat_model(**llm_config)

    fetch_api_node = FetchAPINode(
        input="url", output=["api_events", "content"], node_config=node_config
    )
    state = fetch_api_node.execute(state)

    node_config["schema"] = RegionDetailCollection
    # node_config["schema"] = CarRanking
    node_config["llm_model"] = llm_model
    api_analyzer_node = APIAnalyzerNode(
        input="url & user_prompt & api_events",
        output=["api_analysis", "candidate_api_events"],
        node_config=node_config,
    )
    print(state.keys())
    print(len(state["api_events"]))
    # exit()
    state = api_analyzer_node.execute(state)
    print(state.keys())
    print(state["api_analysis"])
    print(state["candidate_api_events"])
