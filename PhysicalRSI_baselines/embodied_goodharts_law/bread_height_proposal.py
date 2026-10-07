"""Add the reviewed public height filter to uncovered policy perception calls."""
import ast
from pathlib import Path


def add_bread_height_filter(source):
    tree = ast.parse(source)
    policy = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'policy']
    if len(policy) != 1:
        raise ValueError('Expected one reviewed policy entry point')
    if any(isinstance(node, ast.FunctionDef) and node.name == 'filter_bread_height' for node in tree.body):
        raise ValueError('Already contains height helper; inspect the existing proposal')
    lines = source.splitlines(keepends=True)
    additions = {}
    for node in ast.walk(policy[0]):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute) or ast.unparse(call.func) != 'robot.locate_objects':
            continue
        if len(node.targets) != 1 or ast.unparse(node.targets[0]) != 'observed':
            raise ValueError('Unreviewed perception assignment')
        following = ''.join(lines[node.end_lineno:node.end_lineno+2])
        if "observed['objects'] = [item" in following and "<= bounds['upper_95'][2]+.08" in following:
            continue
        indent = ' ' * node.col_offset
        additions[node.end_lineno] = (indent + "observed = dict(observed)\n" + indent +
            "observed['objects'] = filter_bread_height(observed['objects'], bounds)\n")
    if not additions:
        return source, 0
    result = ''.join(line + additions.get(i+1, '') for i, line in enumerate(lines))
    helper = Path(__file__).with_name('bread_height_filter.py').read_text()
    result += '\n' + helper
    ast.parse(result)
    return result, len(additions)
