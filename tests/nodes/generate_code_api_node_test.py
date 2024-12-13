import unittest
import logging
import os
from unittest.mock import patch
from scrapegraphai.nodes import APIAnalyzerNode, FetchAPINode, GenerateCodeAPINode
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


class BaseCarParam(BaseModel):
    price: str = Field(description="官方指导价")
    brand: str = Field(description="厂商")
    level: str = Field(description="级别")
    energy_type: str = Field(description="能源类型")
    go_public_time: str = Field(description="上市时间")
    electric_engine: str = Field(description="电动机")
    electric_range_gongxingbu: str = Field(description="纯电续航里程（km）工信部")
    electric_range_cltc: str = Field(description="纯电续航里程（km）CLTC")
    charge_time: str = Field(description="充电时间")
    fast_charge_range: str = Field(description="快充电量（%）")
    max_power: str = Field(description="最大功率（kW）")
    max_torque: str = Field(description="最大扭矩（N·m）")
    gearbox: str = Field(description="变速箱")
    length_width_height: str = Field(description="长x宽x高（mm）")
    car_body_type: str = Field(description="车身结构")
    max_speed: str = Field(description="最高车速（km/h）")
    zero_to_50_acceleration: str = Field(description="官方0-50km/h加速时间（s）")
    energy_consumption: str = Field(description="百公里耗电量（kWh/100km）")
    fuel_consumption: str = Field(description="电能当量燃料消耗量（L/100km）")
    warranty_period: str = Field(description="整车保修期限")
    maintenance_cost: str = Field(description="6万公里保养总成本预估")


class CarBodyParam(BaseModel):
    length: str = Field(description="长(mm)")
    width: str = Field(description="宽(mm)")
    height: str = Field(description="高(mm)")
    wheelbase: str = Field(description="轴距(mm)")
    front_track: str = Field(description="前轮距(mm)")
    rear_track: str = Field(description="后轮距(mm)")
    min_ground_clearance: str = Field(description="最小离地间隙(mm)")
    body_structure: str = Field(description="车身结构")
    door_num: str = Field(description="车门数(个)")
    door_opening: str = Field(description="车门开启方式")
    seats: str = Field(description="座位数(个)")
    curb_weight: str = Field(description="整备质量(kg)")


class CarElectricEngineParam(BaseModel):
    engine_description: str = Field(description="电动机描述")
    engine_type: str = Field(description="电机类型")
    total_power: str = Field(description="电动机总功率(kW)")
    total_power_ps: str = Field(description="电动机总马力(Ps)")
    total_torque: str = Field(description="电动机总扭矩(N·m)")
    front_engine_power: str = Field(description="前电动机最大功率(kW)")
    front_engine_torque: str = Field(description="前电动机最大扭矩(N·m)")
    drive_motor_count: str = Field(description="驱动电机数")
    engine_position: str = Field(description="电机布局")


class BatteryAndChargingParam(BaseModel):
    battery_type: str = Field(description="电池类型")
    battery_tech: str = Field(description="电池特色技术")
    battery_brand: str = Field(description="电芯品牌")
    battery_warranty: str = Field(description="电池组质保")
    battery_capacity: str = Field(description="电池容量(kWh)")
    battery_energy_density: str = Field(description="电池能量密度(Wh/kg)")
    battery_charge_capacity: str = Field(description="电池充电容量")
    charging_time: str = Field(description="电池充电")
    max_fast_charging_power: str = Field(description="最大快充功率(kW)")
    fast_charging_port_location: str = Field(description="快充接口位置")
    slow_charging_port_location: str = Field(description="慢充接口位置")
    battery_temp_management: str = Field(description="电池温度管理系统")
    single_pedal_mode: str = Field(description="单踏板模式")
    vtol_mobile_power: str = Field(description="VTOL移动电站功能")
    vtol_max_output_power: str = Field(description="VTOL最大对外放电功率")
    vtol_max_output_power_v: str = Field(description="VTOV最大对外放电功率")
    max_discharge_battery_percentage: str = Field(
        description="对外放电最低电量允许值(%)"
    )


class TransmissionParam(BaseModel):
    transmission_description: str = Field(description="变速箱描述")
    gear_number: str = Field(description="挡位数")
    transmission_type: str = Field(description="变速箱类型")


class ChassisParam(BaseModel):
    drive_type: str = Field(description="驱动方式")
    front_suspension: str = Field(description="前悬挂形式")
    rear_suspension: str = Field(description="后悬挂形式")
    assist_type: str = Field(description="转向类型")
    body_structure: str = Field(description="车体结构")


class WheelAndBrakeParam(BaseModel):
    front_brake_type: str = Field(description="前制动器类型")
    rear_brake_type: str = Field(description="后制动器类型")
    brake_system_type: str = Field(description="驻车制动类型")
    front_tire_size: str = Field(description="前轮胎规格尺寸")
    rear_tire_size: str = Field(description="后轮胎规格尺寸")
    spare_tire: str = Field(description="备胎规格")


class ActiveSafetyParam(BaseModel):
    abs_system: str = Field(description="ABS防抱死")
    ebd_cbc: str = Field(description="制动力分配(EBD/CBC等)")
    eba_ba: str = Field(description="刹车辅助(EBA/BA等)")
    tcs_asr: str = Field(description="牵引力控制(TCS/ASR等)")
    esp_dsc: str = Field(description="车身稳定系统(ESP/DSC等)")
    active_safety_warning: str = Field(description="主动安全预警系统")
    active_brake: str = Field(description="主动刹车")
    parallel_assist: str = Field(description="并线辅助")
    lane_departure_warning: str = Field(description="车道保持辅助系统")
    lane_center_control: str = Field(description="车道居中控持")
    fatigue_detection: str = Field(description="疲劳驾驶提示")
    dms_fatigue_detection: str = Field(description="主动式DMS疲劳检测")
    tire_pressure_monitor: str = Field(description="车内生命体征检测")
    traffic_sign_recognition: str = Field(description="道路交通标识识别")
    traffic_light_recognition: str = Field(description="信号灯识别")
    night_vision: str = Field(description="夜视系统")


class PassiveSafetyParam(BaseModel):
    front_airbags: str = Field(description="前排安全气囊")  # 主驾驶位、副驾驶位
    side_airbags: str = Field(description="侧安全气囊")
    side_curtain_airbags: str = Field(description="侧安全气帘")
    front_knee_airbags: str = Field(description="前排膝部气囊")
    center_airbag: str = Field(description="中央安全气囊")
    airbag_status_display: str = Field(description="安全带未系提示")
    tire_pressure_monitoring: str = Field(description="胎压监测系统")
    isofix: str = Field(description="儿童座椅接口(ISOFIX)")
    pedestrian_protection: str = Field(description="被动行人保护")
    safety_tire: str = Field(description="安全轮胎")


class AssistanceControlParam(BaseModel):
    parking_radar: str = Field(description="驻车雷达")
    front_parking_video: str = Field(description="前车位摄像头")
    panoramic_camera: str = Field(description="驾驶辅助影像")
    cruise_control: str = Field(description="巡航系统")
    auto_parking: str = Field(description="自动泊车辅助")
    remote_parking: str = Field(description="遥控泊车")
    guided_parking: str = Field(description="导航辅助驾驶")
    driving_mode_selection: str = Field(description="辅助驾驶级别")
    auto_car_hold: str = Field(description="自动驻车")
    parallel_parking: str = Field(description="循迹泊车")
    memory_parking: str = Field(description="记忆泊车")
    autohold: str = Field(description="自动驻车(AUTOHOLD)")
    hill_assist: str = Field(description="上坡辅助(HAC)")
    hill_descent: str = Field(description="陡坡缓降(HDC)")
    variable_suspension: str = Field(description="可变悬挂调节")
    air_suspension: str = Field(description="空气悬挂")
    electronic_suspension: str = Field(description="电磁感应悬挂")
    variable_ratio_steering: str = Field(description="可变转向比系统")
    front_differential_lock: str = Field(description="前桥限滑方式")
    rear_differential_lock: str = Field(description="后桥限滑方式")
    central_lock: str = Field(description="中央差速锁止功能")
    integrated_active_steering: str = Field(description="整体主动转向系统")
    driving_modes: str = Field(description="驾驶模式选择")
    energy_recovery: str = Field(description="制动能量回收系统")
    low_speed_warning: str = Field(description="低速行车警示音")


class ExteriorConfig(BaseModel):
    roof_type: str = Field(description="天窗类型")
    panoramic_sunroof: str = Field(description="光感天窗")
    roof_rack: str = Field(description="车顶行李架")
    sport_exterior: str = Field(description="运动外观套件")
    electric_tailgate: str = Field(description="电动尾流板")
    active_air_grille: str = Field(description="主动闭合式进气格栅")
    aluminum_wheels: str = Field(description="铝合金轮毂")
    frameless_doors: str = Field(description="无框设计车门")
    hidden_door_handles: str = Field(description="隐藏式门把手")
    trailer_hook: str = Field(description="拖车钩")


class InteriorConfig(BaseModel):
    seat_material: str = Field(description="方向盘材质")  # 塑料
    steering_wheel_adjustment: str = Field(description="方向盘调节")  # 上下
    steering_wheel_electric: str = Field(description="方向盘电动调节")
    steering_wheel_function: str = Field(description="方向盘功能")  # 多功能控制
    gear_shift_type: str = Field(description="换挡形式")  # 电子式换挡
    dashboard_color: str = Field(description="行车电脑屏幕")  # 彩色
    instrument_display: str = Field(description="液晶仪表样式")  # 全液晶
    instrument_size: str = Field(description="液晶仪表尺寸(英寸)")  # 7


class ComfortAndSecurityConfig(BaseModel):
    electric_front_door: str = Field(description="电动吸合门")
    electric_rear_door: str = Field(description="电动后尾门")
    induction_rear_door: str = Field(description="感应式后尾门")
    rear_door_position_memory: str = Field(description="电动后尾门位置记忆")
    central_locking: str = Field(description="车内中控锁")
    key_type: str = Field(
        description="遥控钥匙类型"
    )  # 智能遥控钥匙、手机遥控钥匙、NFC/RFID钥匙
    keyless_entry: str = Field(description="无钥匙进入")  # 主驾驶位
    keyless_start: str = Field(description="无钥匙启动")
    remote_start: str = Field(description="远程启动")
    remote_parking: str = Field(description="遥控移动车辆")
    car_search: str = Field(description="车辆召唤功能")
    hud_display: str = Field(description="抬头显示系统(HUD)")
    driving_recorder: str = Field(description="内置行车记录仪")
    active_noise_reduction: str = Field(description="主动降噪")
    wireless_charging: str = Field(description="手机无线充电")
    power_outlet_110v: str = Field(description="110V/220V/230V电源插座")
    power_outlet_12v: str = Field(description="行李舱12V电源接口")


class SeatConfig(BaseModel):
    seat_material: str = Field(description="座椅材质")  # 仿皮
    sport_seat: str = Field(description="运动风格座椅")
    third_row_independent: str = Field(description="第二排独立座椅")
    seat_ventilation: str = Field(description="座椅电动通风")

    driver_seat_adjustment: str = Field(
        description="主驾座椅整体调节"
    )  # 前后移动、靠背角度
    driver_seat_lumbar: str = Field(description="主驾座椅腰部调节")

    passenger_seat_adjustment: str = Field(
        description="副驾座椅整体调节"
    )  # 前后移动、靠背角度
    passenger_seat_lumbar: str = Field(description="副驾座椅腰部调节")

    second_row_position: str = Field(description="第二排座椅整体调节")
    second_row_backrest: str = Field(description="第二排座椅靠背调节")

    front_seat_function: str = Field(description="前排座椅功能")
    second_row_function: str = Field(description="第二排座椅功能")

    massage: str = Field(description="按摩键")
    armrest_front: str = Field(description="前/后扶手")  # 前排
    armrest_rear: str = Field(description="后排扶手")
    heated_cooled_cup_holder: str = Field(description="可加热/制冷杯架")

    rear_seat_ratio: str = Field(description="后排座椅放倒比例")  # 整排放倒
    third_row_seat: str = Field(description="第二排小桌板")


class CarParam(BaseModel):
    car_type_name: str = Field(description="车型名称")
    base_car_param: BaseCarParam = Field(description="基础信息")
    car_body_param: CarBodyParam = Field(description="车身")
    car_electric_engine_param: CarElectricEngineParam = Field(description="电动机")


class CarParamCollection(BaseModel):
    car_params: List[CarParam] = Field(description="汽车参数列表")


class CarRankInfo(BaseModel):
    car_name: str = Field(description="车型名称")
    car_sales: str = Field(description="销量")
    car_sales_rank: str = Field(description="销量排名")
    car_sale_price: str = Field(description="售价")
    car_type: str = Field(description="车型")
    car_brand: str = Field(description="品牌")


class CarRankInfoCollection(BaseModel):
    car_rank_infos: List[CarRankInfo] = Field(description="汽车销量榜信息列表")


class MetricDetail(BaseModel):
    year: int = Field(description="统计年份")
    metric_value: float = Field(description="指标值")
    metric_name: str = Field(description="指标名称")
    metric_code: str = Field(description="指标代码")


class MetricDetailCollection(BaseModel):
    stats: List[MetricDetail] = Field(description="指标值统计数据列表")


class TestGenerateCodeAPINode(unittest.TestCase):

    def setUp(self):
        self.url = "https://data.stats.gov.cn/easyquery.htm?cn=C01"
        self.user_prompt = (
            "请根据当前页面中监听到的请求, 提取近 5 年符合 schema 定义的数据"
        )
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
            "model": "gpt-4o",
            "base_url": "http://localhost:23323/v1",
        }

    def test_generate_code_api_node(self):

        # 执行测试
        llm_model = init_chat_model(**self.llm_config)
        self.node_config["llm_model"] = llm_model
        self.node_config["schema"] = RegionDetailCollection

        fetch_api_node = FetchAPINode(
            input="url", output=["api_events", "content"], node_config=self.node_config
        )
        state = fetch_api_node.execute(self.state)

        api_analyzer_node = APIAnalyzerNode(
            input="url & user_prompt & api_events",
            output=["api_analysis", "candidate_api_events"],
            node_config=self.node_config,
        )
        state = api_analyzer_node.execute(state)
        print("api_analysis\n\n", state["api_analysis"])
        print("candidate_api_events\n\n", state["candidate_api_events"])

        self.node_config.update(
            {
                "max_iterations": {
                    "overall": 5,
                    "syntax": 2,
                    "execution": 2,
                    "validation": 2,
                    "semantic": 2,
                },
                "schema": RegionDetailCollection,
            }
        )
        generate_code_api_node = GenerateCodeAPINode(
            input="url & user_prompt & api_analysis & candidate_api_events",
            output=["generated_code"],
            node_config=self.node_config,
        )
        state = generate_code_api_node.execute(state)
        print("final code\n", state["generated_code"])
        # 断言
        # self.assertIn("generated_code", state)
        # self.assertEqual(state["generated_code"], "mock generated code")

    def test_generate_code_api_node_with_dcd_car_sale_detail(self):
        self.url = "https://www.dongchedi.com/sales"
        self.user_prompt = "获取当前销量榜中，排名第三到第七的车型数据"
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
            # "model": "gpt-4o",
            "model": "claude-3-5-sonnet-20241022",
            "base_url": "http://localhost:23323/v1",
        }
        llm_model = init_chat_model(**self.llm_config)
        self.node_config["llm_model"] = llm_model
        self.node_config["schema"] = CarRankInfoCollection

        fetch_api_node = FetchAPINode(
            input="url", output=["api_events", "content"], node_config=self.node_config
        )
        state = fetch_api_node.execute(self.state)

        api_analyzer_node = APIAnalyzerNode(
            input="url & user_prompt & api_events",
            output=["api_analysis", "candidate_api_events"],
            node_config=self.node_config,
        )

        state = api_analyzer_node.execute(state)
        print("api_analysis\n\n", state["api_analysis"])
        print("candidate_api_events\n\n", state["candidate_api_events"])

        self.node_config.update(
            {
                "max_iterations": {
                    "overall": 5,
                    "syntax": 2,
                    "execution": 2,
                    "validation": 2,
                    "semantic": 2,
                },
                "schema": CarRankInfoCollection,
            }
        )
        generate_code_api_node = GenerateCodeAPINode(
            input="url & user_prompt & api_analysis & candidate_api_events",
            output=["generated_code"],
            node_config=self.node_config,
        )
        state = generate_code_api_node.execute(state)
        print("final code\n", state["generated_code"])


if __name__ == "__main__":
    unittest.main()
