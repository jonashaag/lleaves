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
    tree_sig = ("T, " * forest.n_args)[:-2]
    tree_names = [f"tree{tree.idx}" for tree in forest.trees]
    tree_funcs = "static tree_func " + ", ".join(tree_names)
    tree_args = ", ".join(f"f[{i}]" for i in range(forest.n_args))
    sum_trees = " + ".join(f"{name}({tree_args})" for name in tree_names)
    yield D(
        f"""
        #include <stdlib.h>
        #define T double
        typedef T tree_func({tree_sig});
        {tree_funcs};
        void forest_root(T *in, T *out, int start, int end) {{
            for (size_t i = start; i < (size_t) end; ++i) {{
                T *f = &in[i * {forest.n_args}];
                out[i] = {sum_trees};
            }}
        }}
        """
    )
    for tree in forest.trees:
        yield from dump_tree(tree)


def dump_tree(tree):
    tree_sig = ", ".join(f"T f{i}" for i in range(len(tree.features)))
    yield f"static T tree{tree.idx}({tree_sig}) {{"
    yield from dump_from(tree.root_node)
    yield "}"


def dump_node(node):
    if node.is_leaf:
        yield f"return (T){node.value};"
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
