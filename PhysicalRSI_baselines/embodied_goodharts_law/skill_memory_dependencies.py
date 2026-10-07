"""Conservative diagnostics for direct memory reads in reviewed primitive code.

This is not a general Python dependency resolver or an execution safety proof.
Callers declare primitive entry points. Literal memory reads in their direct
local call graph are checked, including conditional branches; dynamic calls,
aliases and optional get() reads are not certified by this diagnostic.
"""
import ast


def inspect_skill_memory(source, memory, *, entry_points):
    if not isinstance(memory, dict) or not entry_points:
        raise ValueError('Declare memory and primitive entry points')
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not set(entry_points) <= functions.keys():
        raise ValueError('Primitive entry point is missing')
    visited, reads, pending = set(), {}, list(entry_points)
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for node in ast.walk(functions[name]):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in functions):
                pending.append(node.func.id)
            if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                    and isinstance(node.value, ast.Name) and node.value.id == 'memory'
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                reads.setdefault(node.slice.value, []).append(dict(function=name, line=node.lineno))
    missing = sorted(set(reads) - set(memory))
    return dict(status='missing_direct_memory_keys' if missing else 'no_missing_direct_memory_keys',
                entry_points=sorted(set(entry_points)), visited_functions=sorted(visited),
                missing_keys=missing, direct_reads={key: reads[key] for key in sorted(reads)},
                complete_dependency_analysis=False, execution_validated=False)
