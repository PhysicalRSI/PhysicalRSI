"""Execution-only image inspection, arithmetic and notes. No SDK/file browser."""
import ast
import hashlib
import math
import operator
from pathlib import Path
import uuid

from .contracts import Frame
from .evidence import write_new


def specs():
    def tool(name, description, properties, required):
        return dict(name=name, description=description, parameters=dict(type="object",
            properties=properties, required=required, additionalProperties=False))
    return [tool("inspect_image", "Inspect the current full image or a crop. Pixel box is selected from this observation; never refreshes its timestamp.",
        dict(observation_id=dict(type="string"), camera=dict(type="string", enum=["front", "left", "right"]),
             box=dict(anyOf=[dict(type="null"), dict(type="array", items=dict(type="integer"), minItems=4, maxItems=4)]),
             scale=dict(type="integer", minimum=1, maximum=4)), ["observation_id", "camera", "box", "scale"]),
        tool("calculate", "Pure arithmetic for visual geometry and poses. Scalars/lists, + - * / **, sqrt/sin/cos/atan2/acos/abs/min/max; no files or robot calls.",
             dict(expression=dict(type="string")), ["expression"]),
        tool("notes", "Read or replace same-episode execution working notes. No access to monitor or evaluation.",
             dict(text=dict(anyOf=[dict(type="null"), dict(type="string")])), ["text"]),
        tool("return_to_l1", "Submit accumulated L2 visual experience to generate a reusable L1 candidate and try L1 again.",
             dict(reason=dict(type="string")), ["reason"])]


def calculate(expression):
    if not isinstance(expression, str) or len(expression) > 4096:
        raise ValueError("Arithmetic expression too large")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 256:
        raise ValueError("Arithmetic expression too complex")
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}
    functions = {k: getattr(math, k) for k in ("sqrt", "sin", "cos", "atan2", "acos", "asin", "atan")}
    functions.update(abs=abs, min=min, max=max)
    def visit(n):
        if isinstance(n, ast.Constant) and type(n.value) in (float, int):
            value = float(n.value)
        elif isinstance(n, (ast.List, ast.Tuple)):
            return [visit(v) for v in n.elts]
        elif isinstance(n, ast.Name) and n.id == "pi":
            value = math.pi
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            value = visit(n.operand) * (-1 if isinstance(n.op, ast.USub) else 1)
        elif isinstance(n, ast.BinOp) and type(n.op) in binary:
            a, b = visit(n.left), visit(n.right)
            if not isinstance(a, float) or not isinstance(b, float) or isinstance(n.op, ast.Pow) and abs(b) > 100:
                raise ValueError("Bounded scalar arithmetic required")
            value = binary[type(n.op)](a, b)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in functions and not n.keywords:
            value = functions[n.func.id](*[visit(a) for a in n.args])
        else:
            raise ValueError("Only pure arithmetic is available")
        if not isinstance(value, (float, int)) or not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("Arithmetic result is nonfinite or excessive")
        return float(value)
    return visit(tree.body)


class AnalysisTools:
    def __init__(self):
        self.text = ""

    def invoke(self, name, args, observation, episode):
        if name == "calculate" and set(args) == {"expression"}:
            return dict(value=calculate(args["expression"])), ()
        if name == "notes" and set(args) == {"text"}:
            if args["text"] is not None:
                if not isinstance(args["text"], str) or len(args["text"]) > 16000:
                    raise ValueError("Notes must be a string under 16000 characters")
                self.text = args["text"]
                episode.event("execution_notes", level="L2", text=self.text)
            return dict(text=self.text), ()
        if name != "inspect_image" or set(args) != {"observation_id", "camera", "box", "scale"}:
            raise ValueError("Unknown analysis tool or fields")
        if args["observation_id"] != observation.id:
            raise ValueError("Inspect only this current observation")
        frame = next((f for f in observation.frames if f.camera == args["camera"]), None)
        if frame is None or type(args["scale"]) is not int or not 1 <= args["scale"] <= 4:
            raise ValueError("Invalid camera or scale")
        frame.validate()
        if args["box"] is None:
            return dict(observation_id=observation.id, source=frame.public(), derived=False), (frame,)
        from PIL import Image
        with Image.open(frame.path) as original:
            width, height = original.size
            box = args["box"]
            if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                    or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
                raise ValueError("Crop must lie within the original image")
            scale = args["scale"]
            image = original.crop(box)
            if image.width * image.height * scale**2 > 16_000_000:
                raise ValueError("Requested crop is too large")
            image = image.resize((image.width * scale, image.height * scale))
            target = episode.public / "analysis" / (uuid.uuid4().hex + ".png")
            target.parent.mkdir(exist_ok=True)
            image.save(target)
        derived = Frame(frame.camera, str(target), frame.captured_ns, hashlib.sha256(target.read_bytes()).hexdigest())
        manifest = dict(role="execution", observation_id=observation.id, source=frame.public(),
                        derived=derived.public(), box=box, scale=scale,
                        mapping="source_uv_pixels = crop_pixels / scale + box[:2]",
                        original_size=[width, height], fresh_observation=False)
        write_new(target.with_suffix(".json"), manifest)
        episode.event("image_inspection", level="L2", manifest=manifest)
        return manifest, (derived,)
