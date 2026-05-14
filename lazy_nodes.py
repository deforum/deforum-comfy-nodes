
from .tools import VariantSupport
from .base_node import NODE_POSTFIX, LogicNode


class _AnySwitchInputs(dict):
    def __init__(self, count=5):
        super().__init__((f"any_{i:02d}", ("*",)) for i in range(1, count + 1))

    def __contains__(self, key):
        return isinstance(key, str) and key.startswith("any_")

    def __getitem__(self, key):
        if key in self.keys():
            return super().__getitem__(key)
        if key in self:
            return ("*",)
        raise KeyError(key)


def _any_input_sort_key(key):
    suffix = key.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def _is_empty_any_value(value):
    if value is None:
        return True
    if isinstance(value, dict) and "model" in value and "clip" in value:
        return all(item is None for item in value.values())
    return False


@VariantSupport()
class AnySwitch(LogicNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": _AnySwitchInputs(),
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("*",)
    FUNCTION = "switch"

    def switch(self, **kwargs):
        keys = sorted(
            (key for key in kwargs if key.startswith("any_")),
            key=_any_input_sort_key,
        )
        for key in keys:
            value = kwargs.get(key)
            if not _is_empty_any_value(value):
                return (value,)
        return (None,)


@VariantSupport()
class LazySwitch(LogicNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN",),
                "on_false": ("*", {"lazy": True}),
                "on_true": ("*", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "switch"

    def check_lazy_status(self, switch, on_false = None, on_true = None):
        if switch and on_true is None:
            return ["on_true"]
        if not switch and on_false is None:
            return ["on_false"]

    def switch(self, switch, on_false = None, on_true = None):
        value = on_true if switch else on_false
        return (value,)

NUM_IF_ELSE_NODES = 10
@VariantSupport()
class LazyConditional(LogicNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        args = {
            "value1": ("*", {"lazy": True}),
            "condition1": ("BOOLEAN", {"forceInput": True}),
        }

        for i in range(1,NUM_IF_ELSE_NODES):
            args["value%d" % (i + 1)] = ("*", {"lazy": True})
            args["condition%d" % (i + 1)] = ("BOOLEAN", {"lazy": True, "forceInput": True})

        args["else"] = ("*", {"lazy": True})

        return {
            "required": {},
            "optional": args,
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "conditional"

    def check_lazy_status(self, **kwargs):
        for i in range(0,NUM_IF_ELSE_NODES):
            cond = "condition%d" % (i + 1)
            if cond not in kwargs:
                return [cond]
            if kwargs[cond]:
                val = "value%d" % (i + 1)
                if val not in kwargs:
                    return [val]
                else:
                    return []

        if "else" not in kwargs:
            return ["else"]

    def conditional(self, **kwargs):
        for i in range(0,NUM_IF_ELSE_NODES):
            cond = "condition%d" % (i + 1)
            if cond not in kwargs:
                return [cond]
            if kwargs.get(cond, False):
                val = "value%d" % (i + 1)
                return (kwargs.get(val, None),)

        return (kwargs.get("else", None),)
    
    
@VariantSupport()
class LazyIndexSwitch(LogicNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
                "value0": ("*", {"lazy": True}),
            },
            "optional": {
                "value1": ("*", {"lazy": True}),
                "value2": ("*", {"lazy": True}),
                "value3": ("*", {"lazy": True}),
                "value4": ("*", {"lazy": True}),
                "value5": ("*", {"lazy": True}),
                "value6": ("*", {"lazy": True}),
                "value7": ("*", {"lazy": True}),
                "value8": ("*", {"lazy": True}),
                "value9": ("*", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "index_switch"

    def check_lazy_status(self, index, **kwargs):
        key = "value%d" % index
        if kwargs.get(key, None) is None:
            return [key]

    def index_switch(self, index, **kwargs):
        key = "value%d" % index
        return (kwargs[key],)

@VariantSupport()
class LazyMixImages(LogicNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",{"lazy": True}),
                "image2": ("IMAGE",{"lazy": True}),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mix"

    def check_lazy_status(self, mask, image1 = None, image2 = None):
        mask_min = mask.min()
        mask_max = mask.max()
        needed = []
        if image1 is None and (mask_min != 1.0 or mask_max != 1.0):
            needed.append("image1")
        if image2 is None and (mask_min != 0.0 or mask_max != 0.0):
            needed.append("image2")
        return needed

    # Not trying to handle different batch sizes here just to keep the demo simple
    def mix(self, mask, image1 = None, image2 = None):
        mask_min = mask.min()
        mask_max = mask.max()
        if mask_min == 0.0 and mask_max == 0.0:
            return (image1,)
        elif mask_min == 1.0 and mask_max == 1.0:
            return (image2,)

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(3)
        if mask.shape[3] < image1.shape[3]:
            mask = mask.repeat(1, 1, 1, image1.shape[3])

        return (image1 * (1. - mask) + image2 * mask,)

GENERAL_NODE_CLASS_MAPPINGS = {
    f"AnySwitch{NODE_POSTFIX}": AnySwitch,
    f"LazySwitch{NODE_POSTFIX}": LazySwitch,
    f"LazyIndexSwitch{NODE_POSTFIX}": LazyIndexSwitch,
    f"LazyMixImages{NODE_POSTFIX}": LazyMixImages,
    f"LazyConditional{NODE_POSTFIX}": LazyConditional,
}

GENERAL_NODE_DISPLAY_NAME_MAPPINGS = {
    f"AnySwitch{NODE_POSTFIX}": f"Any Switch {NODE_POSTFIX}",
    f"LazySwitch{NODE_POSTFIX}": f"Lazy Switch {NODE_POSTFIX}",
    f"LazyIndexSwitch{NODE_POSTFIX}": f"Lazy Index Switch {NODE_POSTFIX}",
    f"LazyMixImages{NODE_POSTFIX}": f"Lazy Mix Images {NODE_POSTFIX}",
    f"LazyConditional{NODE_POSTFIX}": f"Lazy Conditional {NODE_POSTFIX}",
}
