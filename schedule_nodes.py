import json
import math
import re

import torch
import torch.nn.functional as F

from .base_node import NODE_POSTFIX, BaseNode
from .tools import VariantSupport


DEFAULT_PROMPT = '''"0" :"",
"11" :"",
"23" :"",
"35" :"",
"47" :"",
"59" :"",
"71" :"",
"83" :"",
"95" :"",
"107" :"",
"119" :""
'''

DEFAULT_VALUE = """0:(0),
11:(0),
23:(0),
35:(0),
47:(0),
59:(0),
71:(0),
83:(0),
95:(0),
107:(0),
119:(0)
"""

_NUMBER_RE = re.compile(r"^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$")
_BACKTICK_RE = re.compile(r"`(.*?)`")

_EVAL_NAMES = {
    name: getattr(math, name)
    for name in (
        "acos",
        "asin",
        "atan",
        "ceil",
        "cos",
        "cosh",
        "exp",
        "fabs",
        "floor",
        "log",
        "log10",
        "pi",
        "pow",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    )
}
_EVAL_NAMES.update({"abs": abs, "max": max, "min": min, "round": round})


class ScheduleNode(BaseNode):
    CATEGORY = "Deforum/Schedule"


def _is_number(value):
    return _NUMBER_RE.match(str(value).strip()) is not None


def _sanitize_value(value):
    return str(value).replace("'", "").replace('"', "").replace("(", "").replace(")", "").strip()


def _eval_expr(expr, *, t=0, max_frames=1, pw_a=0, pw_b=0, pw_c=0, pw_d=0):
    expr = str(expr).strip()
    if _is_number(expr):
        return float(expr)
    env = dict(_EVAL_NAMES)
    env.update(
        {
            "t": t,
            "max_f": max(0, int(max_frames) - 1),
            "pw_a": pw_a,
            "pw_b": pw_b,
            "pw_c": pw_c,
            "pw_d": pw_d,
        }
    )
    return eval(expr, {"__builtins__": {}}, env)


def _value_at(value, *, frame, max_frames, pw_a=0, pw_b=0, pw_c=0, pw_d=0):
    return float(
        _eval_expr(
            _sanitize_value(value),
            t=frame,
            max_frames=max_frames,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
        )
    )


def _frame_key(frame, max_frames):
    frame = _sanitize_value(frame)
    return int(_eval_expr(frame, t=0, max_frames=max_frames))


def _parse_prompt_text(text):
    cleaned = str(text or "").replace("\n", "")
    if not cleaned.strip():
        return {"0": ""}
    wrapped = "{" + cleaned + "}"
    wrapped = re.sub(r",\s*}", "}", wrapped)
    return json.loads(wrapped.strip())


def _parse_value_keyframes(text, max_frames):
    frames = {}
    cleaned = str(text or "").strip().rstrip(",")
    if not cleaned:
        return {0: "0"}
    for part in cleaned.split(","):
        if not part.strip():
            continue
        frame_text, value_text = part.split(":", 1)
        frames[_frame_key(frame_text, max_frames)] = value_text.strip()
    if not frames:
        raise RuntimeError("Key frame string not correctly formatted")
    return frames


def _resolve_prompt_var(value, frame):
    if isinstance(value, (list, tuple)):
        if not value:
            return 0
        return value[min(frame, len(value) - 1)]
    return value


def _eval_prompt_expressions(prompt, frame, max_frames, pw_a=0, pw_b=0, pw_c=0, pw_d=0):
    def replace(match):
        value = _eval_expr(
            match.group(1),
            t=frame,
            max_frames=max_frames,
            pw_a=_resolve_prompt_var(pw_a, frame),
            pw_b=_resolve_prompt_var(pw_b, frame),
            pw_c=_resolve_prompt_var(pw_c, frame),
            pw_d=_resolve_prompt_var(pw_d, frame),
        )
        return str(value)

    return _BACKTICK_RE.sub(replace, str(prompt)).strip()


def _split_prompt_parts(text, pre_text="", app_text=""):
    pos = {}
    neg = {}
    pre_text = str(pre_text or "")
    app_text = str(app_text or "")
    pre_pos, pre_neg = pre_text.split("--neg", 1) if "--neg" in pre_text else (pre_text, "")
    app_pos, app_neg = app_text.split("--neg", 1) if "--neg" in app_text else (app_text, "")

    for frame, prompt in text.items():
        parts = str(prompt).split("--neg", 1)
        positive = parts[0]
        negative = parts[1] if len(parts) > 1 else ""
        pos[frame] = f"{pre_pos} {positive} {app_pos}".strip()
        neg[frame] = f"{pre_neg} {negative} {app_neg}".strip()
    return pos, neg


def _prompt_series(animation_prompts, max_frames, pre_text="", app_text="", pw_a=0, pw_b=0, pw_c=0, pw_d=0):
    max_frames = max(1, int(max_frames))
    parsed = {}
    for key, value in animation_prompts.items():
        parsed[_frame_key(key, max_frames)] = value
    if not parsed:
        parsed[0] = ""

    sorted_prompts = sorted(parsed.items(), key=lambda item: item[0])
    if sorted_prompts[0][0] != 0:
        sorted_prompts.insert(0, (0, sorted_prompts[0][1]))
    if sorted_prompts[-1][0] != max_frames:
        sorted_prompts.append((max_frames, sorted_prompts[-1][1]))

    cur = [""] * max_frames
    nxt = [""] * max_frames
    weights = [1.0] * max_frames

    for (current_key, current_prompt), (next_key, next_prompt) in zip(sorted_prompts, sorted_prompts[1:]):
        if current_key >= next_key:
            continue
        start = max(current_key, 0)
        stop = min(next_key, max_frames)
        step = 1.0 / (next_key - current_key)
        for frame in range(start, stop):
            cur[frame] = str(current_prompt)
            nxt[frame] = str(next_prompt)
            weights[frame] = 1.0 - step * (frame - current_key)

    for frame in range(max_frames):
        cur[frame] = _eval_prompt_expressions(cur[frame], frame, max_frames, pw_a, pw_b, pw_c, pw_d)
        nxt[frame] = _eval_prompt_expressions(nxt[frame], frame, max_frames, pw_a, pw_b, pw_c, pw_d)
    return cur, nxt, weights


def _scheduled_prompts(text, max_frames, pre_text="", app_text="", pw_a=0, pw_b=0, pw_c=0, pw_d=0):
    animation_prompts = _parse_prompt_text(text)
    pos_prompts, neg_prompts = _split_prompt_parts(animation_prompts, pre_text, app_text)
    pos = _prompt_series(pos_prompts, max_frames, pre_text="", app_text="", pw_a=pw_a, pw_b=pw_b, pw_c=pw_c, pw_d=pw_d)
    neg = _prompt_series(neg_prompts, max_frames, pre_text="", app_text="", pw_a=pw_a, pw_b=pw_b, pw_c=pw_c, pw_d=pw_d)
    return pos, neg


def _pad_with_zeros(tensor, target_length):
    current_length = tensor.shape[1]
    if current_length >= target_length:
        return tensor, current_length
    pad_length = target_length - current_length
    left_pad = pad_length // 2
    right_pad = pad_length - left_pad
    return F.pad(tensor, (0, 0, left_pad, right_pad)), target_length


def _conditioning_add_weighted(conditioning_to, conditioning_from, strength):
    out = []
    cond_from = conditioning_from[0][0]
    pooled_from = conditioning_from[0][1].get("pooled_output", None)
    max_size = 0
    for cond_to, meta_to in conditioning_to:
        pooled_to = meta_to.get("pooled_output", pooled_from)
        max_size = max(max_size, cond_to.shape[1], cond_from.shape[1])
        from_padded, _ = _pad_with_zeros(cond_from, max_size)
        to_padded, _ = _pad_with_zeros(cond_to, max_size)
        blended = torch.mul(to_padded, strength) + torch.mul(from_padded, 1.0 - strength)
        meta = meta_to.copy()
        meta["pooled_output"] = pooled_from if pooled_from is not None else pooled_to
        out.append([blended, meta])
    return out


def _encode_prompt(clip, prompt):
    tokens = clip.tokenize(str(prompt))
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


def _pool_prompt_conditioning(clip, current_prompt, next_prompt, weight):
    if str(current_prompt) == str(next_prompt) or weight >= 1.0:
        return _encode_prompt(clip, current_prompt)
    if weight <= 0.0:
        return _encode_prompt(clip, next_prompt)
    return _conditioning_add_weighted(_encode_prompt(clip, current_prompt), _encode_prompt(clip, next_prompt), weight)


def _interpolate_values(text, max_frames):
    max_frames = max(1, int(max_frames))
    keyframes = _parse_value_keyframes(text, max_frames)
    sorted_keys = sorted(keyframes)
    if sorted_keys[0] > 0:
        keyframes[0] = keyframes[sorted_keys[0]]
    if sorted_keys[-1] < max_frames - 1:
        keyframes[max_frames - 1] = keyframes[sorted_keys[-1]]
    sorted_keys = sorted(keyframes)

    values = [0.0] * max_frames
    for index, start in enumerate(sorted_keys):
        end = sorted_keys[index + 1] if index + 1 < len(sorted_keys) else start
        raw_start = keyframes[start]
        raw_end = keyframes[end]
        segment_stop = min(end, max_frames - 1)

        if not _is_number(_sanitize_value(raw_start)):
            for frame in range(max(start, 0), segment_stop + 1):
                values[frame] = _value_at(raw_start, frame=frame, max_frames=max_frames)
            continue

        start_value = _value_at(raw_start, frame=start, max_frames=max_frames)
        end_value = _value_at(raw_end, frame=end, max_frames=max_frames)
        span = max(1, end - start)
        for frame in range(max(start, 0), segment_stop + 1):
            ratio = (frame - start) / span
            values[frame] = start_value + (end_value - start_value) * ratio

    return values


@VariantSupport()
class PromptScheduleNode(ScheduleNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT}),
                "clip": ("CLIP",),
                "max_frames": ("INT", {"default": 120, "min": 1, "max": 999999, "step": 1}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, "forceInput": True}),
                "print_output": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "pre_text": ("STRING", {"multiline": True, "forceInput": True}),
                "app_text": ("STRING", {"multiline": True, "forceInput": True}),
                "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("POS", "NEG")
    FUNCTION = "animate"

    def animate(self, text, max_frames, print_output, current_frame, clip, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text="", app_text=""):
        max_frames = max(1, int(max_frames))
        current_frame = int(current_frame) % max_frames
        (pos_cur, pos_next, pos_weight), (neg_cur, neg_next, neg_weight) = _scheduled_prompts(
            text, max_frames, pre_text, app_text, pw_a, pw_b, pw_c, pw_d
        )
        if print_output:
            print(
                "PromptSchedule:",
                current_frame,
                "POS",
                pos_cur[current_frame],
                "NEXT",
                pos_next[current_frame],
                "WEIGHT",
                pos_weight[current_frame],
            )
        pos = _pool_prompt_conditioning(clip, pos_cur[current_frame], pos_next[current_frame], pos_weight[current_frame])
        neg = _pool_prompt_conditioning(clip, neg_cur[current_frame], neg_next[current_frame], neg_weight[current_frame])
        return (pos, neg)


@VariantSupport()
class StringScheduleNode(ScheduleNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT}),
                "max_frames": ("INT", {"default": 120, "min": 1, "max": 999999, "step": 1}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "print_output": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "pre_text": ("STRING", {"multiline": True, "forceInput": True}),
                "app_text": ("STRING", {"multiline": True, "forceInput": True}),
                "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("POS", "NEG")
    FUNCTION = "animate"

    def animate(self, text, max_frames, current_frame, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text="", app_text="", print_output=False):
        max_frames = max(1, int(max_frames))
        current_frame = int(current_frame) % max_frames
        (pos_cur, _, _), (neg_cur, _, _) = _scheduled_prompts(text, max_frames, pre_text, app_text, pw_a, pw_b, pw_c, pw_d)
        if print_output:
            print("StringSchedule:", current_frame, pos_cur[current_frame])
        return (pos_cur[current_frame], neg_cur[current_frame])


@VariantSupport()
class ValueScheduleNode(ScheduleNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": DEFAULT_VALUE}),
                "max_frames": ("INT", {"default": 120, "min": 1, "max": 999999, "step": 1}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, "forceInput": True}),
                "print_output": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "animate"

    def animate(self, text, max_frames, current_frame, print_output):
        max_frames = max(1, int(max_frames))
        current_frame = int(current_frame) % max_frames
        values = _interpolate_values(text, max_frames)
        value = float(values[current_frame])
        if print_output:
            print("ValueSchedule:", current_frame, value)
        return (value, int(value))


SCHEDULE_NODE_CLASS_MAPPINGS = {
    f"PromptSchedule{NODE_POSTFIX}": PromptScheduleNode,
    f"StringSchedule{NODE_POSTFIX}": StringScheduleNode,
    f"ValueSchedule{NODE_POSTFIX}": ValueScheduleNode,
}

SCHEDULE_NODE_DISPLAY_NAME_MAPPINGS = {
    f"PromptSchedule{NODE_POSTFIX}": f"Prompt Schedule {NODE_POSTFIX}",
    f"StringSchedule{NODE_POSTFIX}": f"String Schedule {NODE_POSTFIX}",
    f"ValueSchedule{NODE_POSTFIX}": f"Value Schedule {NODE_POSTFIX}",
}
