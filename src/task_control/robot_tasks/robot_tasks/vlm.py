"""Independent whole-object VLM detector; no board/door/normal logic."""

import base64, json, math, os, re, stat
from pathlib import Path
import cv2
import requests


def decode(payload, width, height, allowed, max_objects=20):
    if payload.get("status") != "completed":
        raise ValueError("Incomplete VLM response")
    text = "".join(
        c.get("text", "")
        for m in payload.get("output", [])
        if m.get("type") == "message"
        for c in m.get("content", [])
        if c.get("type") == "output_text"
    ).strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        if not match:
            raise ValueError("Invalid VLM JSON fence")
        text = match.group(1)
    data = json.loads(text)
    if (
        not isinstance(data, dict)
        or set(data) != {"objects"}
        or not isinstance(data["objects"], list)
    ):
        raise ValueError("Invalid detection schema")
    if len(data["objects"]) > max_objects:
        raise ValueError("Too many detections")
    results = []
    for item in data["objects"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "bbox_2d"}
            or item["name"] not in allowed
        ):
            raise ValueError("Unexpected object class")
        b = item["bbox_2d"]
        if (
            not isinstance(b, list)
            or len(b) != 4
            or any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                or not 0 <= v <= 1000
                for v in b
            )
        ):
            raise ValueError("Invalid normalized bbox")
        x1, y1, x2, y2 = b
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Inverted bbox")
        box = [
            x1 * width / 1000,
            y1 * height / 1000,
            x2 * width / 1000,
            y2 * height / 1000,
        ]
        if min(box[2] - box[0], box[3] - box[1]) < 6:
            raise ValueError("Detection too small for depth")
        results.append(dict(name=item["name"], bbox=box))
    return results


class Detector:
    def __init__(self, timeout=20.0):
        path = Path.home() / ".config/robot-object-approach/deepseek.json"
        settings = {}
        if path.exists():
            s = path.stat()
            if s.st_uid != os.getuid() or stat.S_IMODE(s.st_mode) & 0o077:
                raise ValueError("DeepSeek config requires owner-only permissions")
            settings = json.loads(path.read_text())
        self.key = os.environ.get("DEEPSEEK_API_KEY") or settings.get("api_key")
        if not self.key:
            raise ValueError("Missing DeepSeek API key")
        self.model = os.environ.get("DEEPSEEK_MODEL") or settings.get(
            "model", "deepseek-flash"
        )
        self.timeout = timeout

    def detect(self, image, description):
        ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("JPEG encoding failed")
        prompt = (
            "请整体理解以下物体描述，在图中检测符合完整描述的所有目标："
            + json.dumps(description, ensure_ascii=False)
            + "。"
            "框出每个目标物体的完整可见部分，不要求正面。每个实例返回一个框。"
            '仅返回JSON：{"objects":[{"name":"类别名称","bbox_2d":[x_min,y_min,x_max,y_max]}]}。'
            "name统一使用给出的完整物体描述。坐标相对整张图归一化为0到1000，x水平、y垂直。"
            '没有这些物体时返回{"objects":[]}。'
        )
        body = dict(
            model=self.model,
            input=[
                dict(
                    role="user",
                    content=[
                        dict(type="input_text", text=prompt),
                        dict(
                            type="input_image",
                            image_url="data:image/jpeg;base64,"
                            + base64.b64encode(jpg).decode(),
                        ),
                    ],
                )
            ],
            reasoning={"effort": "none"},
            max_output_tokens=1800,
            temperature=0,
        )
        with requests.Session() as session:
            session.trust_env = False
            try:
                r = session.post(
                    "https://api.deepseek.com/responses",
                    json=body,
                    headers={"Authorization": "Bearer " + self.key},
                    timeout=(5, self.timeout),
                )
            except requests.RequestException:
                raise ValueError("VLM network request failed") from None
        if r.status_code != 200:
            raise ValueError(f"VLM HTTP {r.status_code}")
        try:
            raw = r.json()
        except ValueError:
            raise ValueError("VLM returned non-JSON response") from None
        return decode(raw, image.shape[1], image.shape[0], [description]), dict(
            provider="deepseek", model=self.model, response_id=raw.get("id")
        )
