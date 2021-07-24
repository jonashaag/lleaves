import functools
import textwrap

from lleaves.compiler.ast.nodes import DecisionNode, Forest, LeafNode, Node, Tree


def dump(obj):
    for type_, func in {
        Forest: dump_forest,
        Tree: dump_tree,
        Node: dump_node,
    }.items():
        if isinstance(obj, type_):
            return func(obj)
    raise TypeError


def format_dump(dump):
    return "\n".join(dump)


def dump_forest(forest):
    tree_args = ", ".join(f"f[{i}]" for i in range(forest.n_args))
    sum_trees = " + ".join(f"tree{tree.idx}({tree_args})" for tree in forest.trees)
    yield D(
        f"""
        module.exports = (in_, out, start, end) => {{
            for (let i = start; i < end; ++i) {{
                const f = in_[i]
                out[i] = {sum_trees}
            }}
        }}
        """
    )
    for tree in forest.trees:
        yield from dump_tree(tree)


def dump_tree(tree):
    tree_sig = ", ".join(f"f{i}" for i in range(len(tree.features)))
    yield f"function tree{tree.idx}({tree_sig}) {{"
    yield from dump_from(tree.root_node)
    yield "}"


def dump_node(node):
    if node.is_leaf:
        yield f"return {node.value}"
    else:
        yield f"if (f{node.split_feature} {node.decision_type} {node.threshold}) {{"
        yield from dump_from(node.left)
        yield "} else {"
        yield from dump_from(node.right)
        yield "}"


def dump_from(node):
    return map(I, dump(node))


def D(s):
    return textwrap.dedent(s).strip()


def I(s):
    return textwrap.indent(s, prefix="  ")
