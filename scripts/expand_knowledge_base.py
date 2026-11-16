"""扩充策略知识库。

阶段四的评测要有意义，知识库必须足够大 —— 否则"Top5 命中率"没有区分度。
本脚本往三张卡表里补充场景（回应策略卡 / 禁止说法卡 / 练习条目）。

用法::

    ./.venv/Scripts/python.exe scripts/expand_knowledge_base.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARDS_DIR = ROOT / "app" / "data" / "cards"

ALL_FITS = ["只想说说", "情绪疏导", "轻量引导", "需要练习"]

# --------------------------------------------------------------------------
# 新增回应策略卡
# --------------------------------------------------------------------------
NEW_REPLY_CARDS = [
    {
        "id": "scene_exam_progress",
        "scene": "exam_progress",
        "title": "考试进步 / 被表扬后的自我怀疑",
        "must": ["具体地复述进步的事实", "点出韧性", "只问一个问题"],
        "steps": ["具体化", "优势强化"],
        "applicable_fit": ["轻量引导", "情绪疏导"],
        "applicable_distortions": ["以偏概全", "读心术"],
        "forbidden_when": ["高危预警"],
        "text": "场景: 考试进步、被表扬，但学生不敢相信或怕别人期待变高。适用: intervention_fit = 轻量引导。必须: 具体地复述进步的事实 → 点出韧性 → 只问一个问题。要点: 不空泛地说\"很棒\"，要复述他说的具体事实；不立刻把期望抬到更高。",
    },
    {
        "id": "scene_subject_pressure",
        "scene": "subject_pressure",
        "title": "选科压力 / 文理分科摇摆",
        "must": ["共情接纳", "不替他做决定", "点出韧性", "只问一个问题"],
        "steps": ["共情接纳", "利弊澄清", "优势强化"],
        "applicable_fit": ["轻量引导", "情绪疏导", "需要练习"],
        "applicable_distortions": ["非黑即白", "灾难化"],
        "forbidden_when": ["高危预警"],
        "text": "场景: 选科、文理分科、要不要学某个科目。适用: intervention_fit = 轻量引导 / 需要练习。必须: 共情接纳 → 不替他做决定 → 点出韧性 → 只问一个问题。要点: 先把\"必须选 X\"的绝对感拆开，再谈他自己看重什么。",
    },
    {
        "id": "scene_sports_failure",
        "scene": "sports_failure",
        "title": "体育 / 比赛挫败",
        "must": ["共情接纳", "把一次结果和长期能力分开", "点出韧性", "只问一个问题"],
        "steps": ["共情接纳", "认知解离", "优势强化"],
        "applicable_fit": ["轻量引导", "情绪疏导"],
        "applicable_distortions": ["以偏概全", "过度自责"],
        "forbidden_when": ["高危预警"],
        "text": "场景: 体育测试没达标、比赛输了、被人说不行。适用: intervention_fit = 轻量引导。必须: 共情接纳 → 把一次结果和长期能力分开 → 点出韧性 → 只问一个问题。要点: 不把身体和运动能力混为一谈。",
    },
    {
        "id": "scene_appearance",
        "scene": "appearance",
        "title": "外貌与身材焦虑",
        "must": ["共情接纳", "不评价外貌", "点出韧性", "只问一个问题"],
        "steps": ["共情接纳", "自我关怀", "优势强化"],
        "applicable_fit": ["轻量引导", "情绪疏导", "需要练习"],
        "applicable_distortions": ["以偏概全", "读心术"],
        "forbidden_when": ["高危预警"],
        "text": "场景: 嫌自己胖、丑、个子矮，被外貌评价困扰。适用: intervention_fit = 轻量引导 / 需要练习。必须: 共情接纳 → 不评价外貌 → 点出韧性 → 只问一个问题。要点: 不说\"你这样也挺好看的\"这种空泛安慰。",
    },
    {
        "id": "scene_friend_leaving",
        "scene": "friend_leaving",
        "title": "好友转学 / 离别",
        "must": ["共情接纳", "不急着开解", "点出韧性", "只问一个问题"],
        "steps": ["共情接纳", "在场感", "优势强化"],
        "applicable_fit": ["只想说说", "情绪疏导", "轻量引导"],
        "applicable_distortions": ["灾难化", "读心术"],
        "forbidden_when": ["高危预警"],
        "text": "场景: 好朋友转学、毕业分开、关系断掉。适用: intervention_fit = 只想说说 / 情绪疏导。必须: 共情接纳 → 不急着开解 → 点出韧性 → 只问一个问题。要点: 离别里的失落需要先被允许存在。",
    },
    {
        "id": "scene_club_competition",
        "scene": "club_competition",
        "title": "社团竞争 / 名额落选",
        "must": ["共情接纳", "确认事实", "点出韧性", "只问一个问题"],
        "steps": ["共情接纳", "事实澄清", "优势强化"],
        "applicable_fit": ["轻量引导", "情绪疏导"],
        "applicable_distortions": ["读心术", "以偏概全"],
        "forbidden_when": ["高危预警"],
        "text": "场景: 社团没选上、比赛名额被占。适用: intervention_fit = 轻量引导。必须: 共情接纳 → 确认事实 → 点出韧性 → 只问一个问题。要点: 落选原因往往还没确定，别替对方补一个。",
    },
    {
        "id": "scene_teacher_misunderstanding",
        "scene": "teacher_misunderstanding",
        "title": "被老师误解",
        "must": ["共情接纳", "不评判老师", "点出韧性", "只问一个问题"],
        "steps": ["共情接纳", "立场中立", "优势强化"],
        "applicable_fit": ["轻量引导", "情绪疏导"],
        "applicable_distortions": ["读心术", "非黑即白"],
        "forbidden_when": ["高危预警"],
        "text": "场景: 被老师误解、冤枉、批评不公。适用: intervention_fit = 轻量引导。必须: 共情接纳 → 不评判老师 → 点出韧性 → 只问一个问题。要点: 站在学生感受一边，但不要煽动对立。",
    },
    {
        "id": "scene_dormitory",
        "scene": "dormitory",
        "title": "宿舍关系",
        "must": ["共情接纳", "不替学生做决定", "点出韧性", "只问一个问题"],
        "steps": ["共情接纳", "事实澄清", "优势强化"],
        "applicable_fit": ["轻量引导", "情绪疏导", "需要练习"],
        "applicable_distortions": ["以偏概全", "读心术"],
        "forbidden_when": ["高危预警"],
        "text": "场景: 室友作息、卫生、借东西、被打扰。适用: intervention_fit = 轻量引导 / 需要练习。必须: 共情接纳 → 不替学生做决定 → 点出韧性 → 只问一个问题。要点: 日常小摩擦不要上纲上线，也不要劝学生\"别计较\"。",
    },
    {
        "id": "scene_parents_control",
        "scene": "parents_control",
        "title": "父母管控 / 手机与行踪",
        "must": ["共情接纳", "不评判任何一方", "点出韧性", "只问一个问题"],
        "steps": ["共情接纳", "立场中立", "优势强化"],
        "applicable_fit": ["轻量引导", "情绪疏导", "需要练习"],
        "applicable_distortions": ["非黑即白", "以偏概全"],
        "forbidden_when": ["高危预警", "学生说只想说说"],
        "text": "场景: 父母查手机、限制外出、不让参加活动。适用: intervention_fit = 轻量引导 / 需要练习。必须: 共情接纳 → 不评判任何一方 → 点出韧性 → 只问一个问题。要点: 理解双方处境的同时，不劝学生\"你就听话吧\"。",
    },
    {
        "id": "scene_procrastination",
        "scene": "procrastination",
        "title": "拖延与时间管理",
        "must": ["共情接纳", "不贴\"懒\"的标签", "点出韧性", "只问一个问题"],
        "steps": ["共情接纳", "事实澄清", "优势强化"],
        "applicable_fit": ["轻量引导", "需要练习", "情绪疏导"],
        "applicable_distortions": ["过度自责", "以偏概全"],
        "forbidden_when": ["高危预警", "学生说只想说说"],
        "text": "场景: 作业拖到 deadline、计划做不完、拖延到焦虑。适用: intervention_fit = 轻量引导 / 需要练习。必须: 共情接纳 → 不贴\"懒\"的标签 → 点出韧性 → 只问一个问题。要点: 拖延背后常是任务太大或太模糊，先问清楚是哪一种。",
    },
]

# --------------------------------------------------------------------------
# 新增禁止说法卡
# --------------------------------------------------------------------------
NEW_BAN_CARDS = [
    {
        "id": "ban_exam_progress",
        "scene": "exam_progress",
        "title": "进步场景禁止说法",
        "bans": ["别骄傲", "这有什么难的", "你运气好而已"],
        "text": "禁止: 别骄傲、这有什么难的、你运气好而已。要点: 否定进步、淡化学生付出、把结果归因于外因。",
    },
    {
        "id": "ban_subject_pressure",
        "scene": "subject_pressure",
        "title": "选科场景禁止说法",
        "bans": ["你必须选理科", "别犹豫了", "选文科没前途"],
        "text": "禁止: 你必须选理科、别犹豫了、选文科没前途。要点: 替学生做决定、放大焦虑、给出没有依据的结论。",
    },
    {
        "id": "ban_appearance",
        "scene": "appearance",
        "title": "外貌焦虑场景禁止说法",
        "bans": ["你这样也挺好的", "想太多了吧", "现在流行这样"],
        "text": "禁止: 你这样也挺好的、想太多了吧、现在流行这样。要点: 空泛安慰、否定感受、拿流行标准压学生。",
    },
    {
        "id": "ban_procrastination",
        "scene": "procrastination",
        "title": "拖延场景禁止说法",
        "bans": ["你就是懒", "自制力差的人没救", "早就该做完了"],
        "text": "禁止: 你就是懒、自制力差的人没救、早就该做完了。要点: 贴人格标签、否定能力、事后指责。",
    },
    {
        "id": "ban_dormitory",
        "scene": "dormitory",
        "title": "宿舍关系禁止说法",
        "bans": ["别计较了", "这种事天天有", "你忍一忍就过去了"],
        "text": "禁止: 别计较了、这种事天天有、你忍一忍就过去了。要点: 正常化学生的处境、要求学生单方面承受。",
    },
]

# --------------------------------------------------------------------------
# 新增练习条目
# --------------------------------------------------------------------------
NEW_EXERCISE_CARDS = [
    {
        "id": "body_scan",
        "scene": "appearance",
        "name": "身体扫描",
        "duration": "6 分钟",
        "steps": [
            "把注意力从上到下放在身体上",
            "不用改变它，只是注意它",
            "注意到紧绷的地方，在心里说\"我注意到你了\"",
            "做三个自然的呼吸",
            "看看感觉有没有变一点",
        ],
        "do_not_use_when": ["高危预警"],
        "text": "场景: 身体紧绷、外貌焦虑引发的躯体反应。时长: 6 分钟。步骤: 注意力从上到下 → 不改变只注意 → 对紧绷处打招呼 → 三个呼吸 → 再看看。不要用在: 高危预警。",
    },
    {
        "id": "thought_record",
        "scene": "self_doubt",
        "name": "想法记录表",
        "duration": "10 分钟",
        "steps": [
            "写下让你难受的那个念头",
            "写下支持这个念头的事实",
            "写下反对这个念头的事实",
            "写一句更贴近事实的说法",
            "把这句话记进优势清单里",
        ],
        "do_not_use_when": ["高危预警", "学生说只想说说"],
        "text": "场景: 自动化负面念头反复出现。时长: 10 分钟。步骤: 写下念头 → 支持事实 → 反对事实 → 换个说法 → 存档。不要用在: 高危预警、学生说只想说说。",
    },
    {
        "id": "pros_cons",
        "scene": "subject_pressure",
        "name": "利弊清单",
        "duration": "12 分钟",
        "steps": [
            "写下两个你真正在意的方向",
            "给每个方向列三条你看重的东西",
            "给每个方向列一条你担心的事",
            "标出哪一条是你自己真正在意的",
            "先不下结论，只把清单留着",
        ],
        "do_not_use_when": ["高危预警"],
        "text": "场景: 选择摇摆、觉得\"必须选一个\"。时长: 12 分钟。步骤: 列方向 → 各自看重的 → 各自担心的 → 标出真正在意的 → 暂不下结论。不要用在: 高危预警。",
    },
    {
        "id": "future_self",
        "scene": "procrastination",
        "name": "未来的自己",
        "duration": "8 分钟",
        "steps": [
            "想象一周后的自己，最想要的状态是什么",
            "写下那个状态下的自己会做的第一个动作",
            "把那个动作缩到 5 分钟以内",
            "今天就只做这 5 分钟",
        ],
        "do_not_use_when": ["高危预警", "学生说只想说说"],
        "text": "场景: 拖延、启动困难。时长: 8 分钟。步骤: 想象一周后的自己 → 写下第一个动作 → 缩到 5 分钟 → 今天就做这 5 分钟。不要用在: 高危预警、学生说只想说说。",
    },
    {
        "id": "value_sort",
        "scene": "subject_pressure",
        "name": "价值排序",
        "duration": "10 分钟",
        "steps": [
            "写下此刻你在意的 5 件事",
            "给它们排序",
            "圈出排在最前面的两件",
            "再看一次：你现在的选择在支持哪一件",
        ],
        "do_not_use_when": ["高危预警"],
        "text": "场景: 不知道自己想要什么。时长: 10 分钟。步骤: 写 5 件在意的 → 排序 → 圈出前两件 → 对照当前选择。不要用在: 高危预警。",
    },
    {
        "id": "sleep_wind_down",
        "scene": "family_conflict",
        "name": "睡前收尾三步",
        "duration": "10 分钟",
        "steps": [
            "把今天还挂着的三件事各写一行",
            "每件事后面写一句\"明天再处理\"",
            "把明天的第一件事写在最上面",
        ],
        "do_not_use_when": ["高危预警"],
        "text": "场景: 晚上躺下后反复回想白天的事。时长: 10 分钟。步骤: 列三件 → 标注明天再处理 → 写出明天第一件事。不要用在: 高危预警。",
    },
    {
        "id": "social_recharge",
        "scene": "loneliness",
        "name": "社交电量恢复",
        "duration": "1 天",
        "steps": [
            "找出一个你愿意待着的集体或活动",
            "这周去一次，不需要说话也没关系",
            "记录去之前和之后的电量差别",
        ],
        "do_not_use_when": ["高危预警"],
        "text": "场景: 长期社交耗竭、觉得没人可以说话。时长: 1 天（一次尝试）。步骤: 找一个愿意待着的场合 → 这周去一次 → 记录前后差别。不要用在: 高危预警。",
    },
    {
        "id": "study_block_25",
        "scene": "procrastination",
        "name": "25 分钟专注块",
        "duration": "25 分钟",
        "steps": [
            "只挑一个最小任务写下来",
            "设 25 分钟计时，只做这一件事",
            "时间到就停，不追加",
            "记录实际用掉的时间",
        ],
        "do_not_use_when": ["高危预警"],
        "text": "场景: 任务太大导致无法启动。时长: 25 分钟。步骤: 挑最小任务 → 计时只做这一件 → 到点就停 → 记录用时。不要用在: 高危预警。",
    },
]


def _write(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{path.name}: {len(data['cards'])} 张卡")


def main() -> int:
    reply_path = CARDS_DIR / "student_reply_cards.json"
    reply = json.loads(reply_path.read_text(encoding="utf-8"))
    existing = {c["id"] for c in reply["cards"]}
    added = 0
    for card in NEW_REPLY_CARDS:
        if card["id"] in existing:
            continue
        reply["cards"].append(card)
        added += 1
    _write(reply_path, reply)
    print(f"  新增回应策略卡 {added} 张")

    ban_path = CARDS_DIR / "ban_cards.json"
    ban = json.loads(ban_path.read_text(encoding="utf-8"))
    existing = {c["id"] for c in ban["cards"]}
    added = 0
    for card in NEW_BAN_CARDS:
        if card["id"] in existing:
            continue
        card["applicable_fit"] = ALL_FITS
        card["applicable_distortions"] = []
        ban["cards"].append(card)
        added += 1
    _write(ban_path, ban)
    print(f"  新增禁止说法卡 {added} 张")

    ex_path = CARDS_DIR / "exercise_cards.json"
    ex = json.loads(ex_path.read_text(encoding="utf-8"))
    existing = {c["id"] for c in ex["cards"]}
    added = 0
    for card in NEW_EXERCISE_CARDS:
        if card["id"] in existing:
            continue
        card["applicable_fit"] = ["需要练习"]
        ex["cards"].append(card)
        added += 1
    _write(ex_path, ex)
    print(f"  新增练习条目 {added} 张")
    return 0


if __name__ == "__main__":
    sys.exit(main())
