import unittest
import os
from unittest.mock import patch
from scrapegraphai.nodes.fetch_api_node import FetchAPINode
from scrapegraphai.nodes.api_diagnose_node import APIDiagnoseNode
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List


import agentops
from dotenv import load_dotenv

load_dotenv()

AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")

agentops.init(
    api_key=os.getenv("AGENTOPS_API_KEY"),
)


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


class GDPDetail(BaseModel):
    year: int = Field(description="统计年份")
    gdp: float = Field(description="国内生产总值（亿元）")
    domestic_total_income: float = Field(description="国民总收入（亿元）")
    first_industry_gdp: float = Field(description="第一产业增加值（亿元）")
    second_industry_gdp: float = Field(description="第二产业增加值（亿元）")
    tertiary_industry_gdp: float = Field(description="第三产业增加值（亿元）")
    per_capita_gdp: float = Field(description="人均国内生产总值（元）")


class GDPDetailCollection(BaseModel):
    stats: List[GDPDetail] = Field(description="国内生产总值统计数据列表")


class IndustryDetail(BaseModel):
    year: int = Field(description="统计年份")
    enterprise_number: str = Field(description="工业企业单位数")


class IndustryDetailCollection(BaseModel):
    stats: List[IndustryDetail] = Field(description="工业企业单位数统计数据列表")


class MetricDetail(BaseModel):
    year: int = Field(description="统计年份")
    metric_value: float = Field(description="指标值")
    metric_name: str = Field(description="指标名称")
    metric_code: str = Field(description="指标代码")


class MetricDetailCollection(BaseModel):
    stats: List[MetricDetail] = Field(description="指标值统计数据列表")


class TestAPIDiagnoseNode(unittest.TestCase):

    def setUp(self):
        self.url = "https://data.stats.gov.cn/easyquery.htm?cn=C01"
        self.user_prompt = "请根据当前页面中监听到的请求, 帮我生成相关数据抓取代码，获取最近5年所有工业分支指标下所有子指标信息，并以如下 schema 形式输出。"

        self.state = {
            "url": self.url,
            "user_prompt": self.user_prompt,
        }
        self.node_config = {
            "verbose": True,
            "headless": False,
        }
        self.llm_config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model_provider": "openai",
            # "model": "gpt-4o-mini",
            # "model": "claude-3-5-sonnet-20241022",
            "model": "gpt-4o",
            "base_url": "http://localhost:23323/v1",
        }

    def test_generate_code_api_node(self):
        # 执行测试
        llm_model = init_chat_model(**self.llm_config)
        self.node_config["llm_model"] = llm_model
        self.node_config["schema"] = MetricDetailCollection

        fetch_api_node = FetchAPINode(
            input="url", output=["api_events", "content"], node_config=self.node_config
        )
        state = fetch_api_node.execute(self.state)

        api_diagnose_node = APIDiagnoseNode(
            input="url & user_prompt & api_events",
            output=["api_calling_strategy", "candidate_api_events", "reference_code"],
            node_config=self.node_config,
        )
        state = api_diagnose_node.execute(state)


if __name__ == "__main__":
    unittest.main()
