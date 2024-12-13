import json
import asyncio
from scrapegraphai.nodes import FetchAPINode
from scrapegraphai.nodes.fetch_api_node import capture_api_events


def test_fetch_api_call_request():
    fetch_api_node = FetchAPINode(
        input="url", output=["api_events", "content"], node_config={"headless": True}
    )

    # Execute node and simulate events
    result = fetch_api_node.execute(
        {"url": "https://data.stats.gov.cn/easyquery.htm?cn=C01"}
    )
    print(len(result["api_events"]))
    print(len(result["content"]))


def test_capture_api_events():
    content, api_events = asyncio.run(
        capture_api_events("https://www.dongchedi.com/sales", headless=True)
    )
    assert content is not None
    assert api_events is not None
    assert len(api_events) > 0
    print(json.dumps(api_events[-1]))
