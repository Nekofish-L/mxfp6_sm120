#!/usr/bin/env python3
"""Compare every CUDA function's SASS and resource metadata in two libraries."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def inspect(path):
    tool = shutil.which("cuobjdump")
    if tool is None:
        raise RuntimeError("cuobjdump must be on PATH")
    result = subprocess.run([tool, "--dump-sass", str(path)],check=True,capture_output=True,text=True)
    functions = {}
    for block in result.stdout.split("Function :")[1:]:
        name = block.splitlines()[0].strip()
        # Preserve instruction addresses, operands and encodings. Ignore only
        # object-file headers separating functions from different fatbins.
        # cuobjdump pads columns according to the longest instruction in the
        # entire input; a one-kernel CUBIN and a full DSO can differ in padding.
        lines = [" ".join(line.split()) for line in block.splitlines()[1:]
                 if re.match(r"\s*/\*",line)]
        digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
        functions.setdefault(name, []).append(digest)
    if not functions:
        raise RuntimeError(f"no CUDA functions found in {path}")
    resources = subprocess.run([tool,"--dump-resource-usage",str(path)],check=True,capture_output=True,text=True)
    # cuobjdump's object header includes paths; only compare function resources.
    resource_lines = [line.strip() for line in resources.stdout.splitlines()
                      if re.match(r"\s*(Function|REG:|SHARED:|LOCAL:|STACK:)",line)]
    resource_functions = {}
    name = None
    for line in resource_lines:
        if line.startswith('Function '):
            name = line[len('Function '):].rstrip(':')
        elif name is not None:
            resource_functions.setdefault(name, []).append(line)
    return dict(library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                functions={k: sorted(v) for k,v in functions.items()},
                resource_functions={k: sorted(v) for k,v in resource_functions.items()},
                resource_sha256=hashlib.sha256("\n".join(resource_lines).encode()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("before",type=Path)
    p.add_argument("after",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--allow-subset",action="store_true",help="Permit removed kernels, but require all retained kernels/resources to match")
    a=p.parse_args()
    before,after=inspect(a.before),inspect(a.after)
    names = after['functions'].keys() if a.allow_subset else before['functions'].keys() | after['functions'].keys()
    changed=[name for name in sorted(names)
             if before['functions'].get(name) != after['functions'].get(name)]
    resources_equal = (all(before['resource_functions'].get(name)==after['resource_functions'].get(name) for name in after['resource_functions'])
                       if a.allow_subset else before['resource_functions']==after['resource_functions'])
    report=dict(before=before,after=after,changed_functions=changed,
                removed_functions=sorted(before['functions'].keys()-after['functions'].keys()),
                resources_equal=resources_equal)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(f"CUDA functions: {len(before['functions'])} -> {len(after['functions'])}; changed: {len(changed)}; resources equal: {report['resources_equal']}")
    if changed or not report['resources_equal']:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
