from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_c_sharp
import tree_sitter_java
from tree_sitter import Language, Node, Parser


EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".explorer",
    ".codegraph",
    ".idea",
    ".vs",
    "node_modules",
    "bin",
    "obj",
    "build",
    "target",
    "packages",
    "library",
    "temp",
}

FUNCTION_TYPES = {
    "method_declaration",
    "constructor_declaration",
    "compact_constructor_declaration",
    "local_function_statement",
}

CONTAINER_TYPES = {
    "class_declaration",
    "interface_declaration",
    "struct_declaration",
    "record_declaration",
    "enum_declaration",
    "record_struct_declaration",
}

PARSERS = {
    ".cs": Parser(Language(tree_sitter_c_sharp.language())),
    ".java": Parser(Language(tree_sitter_java.language())),
}


@dataclass(frozen=True)
class Symbol:
    name: str
    code: str
    start_line: int
    end_line: int


def _raise_walk_error(error: OSError) -> None:
    raise error


def find_code_files(root: Path) -> list[Path]:
    files = []

    for directory, subdirs, names in os.walk(
        root,
        followlinks=False,
        onerror=_raise_walk_error,
    ):
        current = Path(directory)

        subdirs[:] = sorted(
            name
            for name in subdirs
            if name.lower() not in EXCLUDED_DIRS
            and not (current / name).is_symlink()
        )

        for name in sorted(names):
            path = current / name
            if path.suffix.lower() in PARSERS and not path.is_symlink():
                files.append(path)

    return sorted(files)


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8")


def parse_file(path: Path) -> tuple[str, str, list[Symbol]]:
    raw = path.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()

    # Поддерживаем UTF-8, в том числе с BOM. Другую кодировку не подменяем.
    text = raw.decode("utf-8-sig")
    source = text.encode("utf-8")

    tree = PARSERS[path.suffix.lower()].parse(source)

    if tree.root_node.has_error:
        raise ValueError(
            f"Tree-sitter could not fully parse {path}. "
            "Check syntax and grammar support; the file was not silently skipped."
        )

    symbols = []
    stack = [tree.root_node]

    while stack:
        node = stack.pop()
        stack.extend(reversed(node.named_children))

        if node.type not in FUNCTION_TYPES:
            continue

        name_node = node.child_by_field_name("name")
        if name_node is None:
            raise ValueError(
                f"Cannot identify {node.type} in {path}:"
                f"{node.start_point.row + 1}"
            )

        name = _node_text(name_node, source)
        containers = []

        parent = node.parent
        while parent is not None:
            if parent.type in CONTAINER_TYPES | FUNCTION_TYPES:
                parent_name = parent.child_by_field_name("name")
                if parent_name is not None:
                    containers.append(_node_text(parent_name, source))
            parent = parent.parent

        qualified_name = ".".join([*reversed(containers), name])

        parameters = node.child_by_field_name("parameters")
        if parameters is not None:
            qualified_name += _node_text(parameters, source)

        start_line = node.start_point.row + 1
        end_line = node.end_point.row + (1 if node.end_point.column else 0)

        symbols.append(
            Symbol(
                name=qualified_name,
                code=_node_text(node, source),
                start_line=start_line,
                end_line=max(start_line, end_line),
            )
        )

    return text, source_hash, symbols
