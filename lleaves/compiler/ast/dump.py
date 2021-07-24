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
    yield f"Forest n_features={forest.n_args} objective={forest.objective_func}({forest.objective_func_config})"
    for tree in forest.trees:
        yield from dump_from(tree)


def dump_tree(tree):
    yield f"Tree idx={tree.idx} n_features={len(tree.features)}"
    yield from dump_from(tree.root_node)


def dump_node(node):
    if node.is_leaf:
        yield str(node.value)
    else:
        yield f"if ${node.split_feature} {node.decision_type} {node.threshold}"
        yield "then"
        yield from dump_from(node.left)
        yield "else"
        yield from dump_from(node.right)


def dump_from(node):
    return map(I, dump(node))


def D(s):
    return textwrap.dedent(s).strip()


def I(s):
    return textwrap.indent(s, prefix="  ")
