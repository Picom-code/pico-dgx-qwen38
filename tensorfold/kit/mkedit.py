#!/usr/bin/env python3
# mkedit.py OUTDIR : two prompt sets for the copy-draft measurements, built from Python's own standard library sources on this box
# (deterministic): edit.jsonl (rewrite-a-file tasks: the whole file comes back with a small change) and agent.jsonl (agent-style turns:
# a tool schema, earlier tool output with file contents, and an instruction that ends in a tool call carrying file text or a patch).
import sys, os, json, sysconfig
out = sys.argv[1]; os.makedirs(out, exist_ok=True); lib = sysconfig.get_paths()["stdlib"]
names = ["textwrap", "bisect", "heapq", "colorsys", "fnmatch", "glob", "shlex", "stat", "string", "copy", "keyword", "linecache", "reprlib", "sched", "queue", "numbers",
         "abc", "contextlib", "functools", "getopt", "graphlib", "netrc", "pprint", "quopri", "this", "token", "types", "uu", "weakref", "nturl2path", "opcode", "operator"]
files = []
for n in names:
    p = os.path.join(lib, n + ".py")
    if not os.path.exists(p): continue
    src = open(p, encoding="utf-8", errors="replace").read().split("\n")
    # a slice of ~70-110 lines that starts at a def/class if there is one
    st = next((i for i, l in enumerate(src) if l.startswith(("def ", "class "))), 0)
    body = "\n".join(src[st:st + 70 + 5 * (len(files) % 9)])
    if len(body) > 800: files.append((n, body))
tasks = ["Add a one-line comment above every function or method definition saying what it does. Return the complete file, nothing else.",
         "Rename the first function or class defined in this file by appending _v2 to its name, everywhere it is used. Return the complete updated file, nothing else.",
         "Add type hints to the function signatures where the types are obvious. Do not change anything else. Return the complete file, nothing else.",
         "Replace every use of single-quoted string literals with double-quoted ones where that is safe. Return the complete file, nothing else."]
with open(os.path.join(out, "edit.jsonl"), "w") as f:
    for i, (n, body) in enumerate(files[:24]):
        f.write(json.dumps({"id": i, "cat": "edit", "prompt": f"Here is the file {n}.py:\n\n```python\n{body}\n```\n\n{tasks[i % len(tasks)]}"}) + "\n")
tools = ('You are a coding agent. You have these tools, called by replying with one JSON object and nothing else:\n'
         '{"tool": "read_file", "path": str}\n{"tool": "write_file", "path": str, "content": str}\n{"tool": "apply_patch", "path": str, "old": str, "new": str}\n{"tool": "run", "cmd": str}\n')
asks = ["The task: add a module-level constant DEBUG = False near the top and make the first function print its arguments when DEBUG is true. Reply with the write_file call holding the whole new file.",
        "The task: the first function needs a docstring that explains its arguments. Reply with one apply_patch call; `old` must be the exact current text of that function's first three lines.",
        "The task: summarise in five bullet points what this file does, quoting the exact signature line of each function you mention.",
        "The task: write a pytest file tests/test_{n}.py with three tests for the first function, then reply with the write_file call for it."]
with open(os.path.join(out, "agent.jsonl"), "w") as f:
    for i, (n, body) in enumerate(files[:16]):
        hist = (f'{tools}\n[user] Please work on src/{n}.py.\n[assistant] {{"tool": "read_file", "path": "src/{n}.py"}}\n[tool result]\n{body}\n'
                f'[assistant] {{"tool": "run", "cmd": "python -m pyflakes src/{n}.py"}}\n[tool result]\n(no output)\n[user] {asks[i % len(asks)].replace("{n}", n)}')
        f.write(json.dumps({"id": i, "cat": "agent", "prompt": hist}) + "\n")
print("edit", min(24, len(files)), "agent", min(16, len(files)), "files from", lib)
