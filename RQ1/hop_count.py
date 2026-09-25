"""
Script that recomputes RQ1 hop count

Definitions:
- node: unique by table name + alias
- subquery: WHERE IN, EXISTS, FROM derived table, ANY/ALL are included,
  scalar subquery is excluded
- edge: connected by an = condition (whether from JOIN ON or WHERE)
- hop_count = graph diameter (if disconnected, use the diameter of the biggest
  chunk and flag it for manual review)
- hop_naive (count of the string "JOIN") and hop_breadth (|nodes| - 1) are
  recorded too
- subquery-related stuff gets its own column
"""

from pathlib import Path
import json
import networkx as nx
import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope, ScopeType


# path setup
BASE = Path(__file__).parent.parent
DEV_JSON = BASE / "dev.json"
OUT_CSV = Path(__file__).parent / "financial_hop_count_results.csv"
DB_ID = "financial"


def load_questions(path, db_id):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    res = []
    for row in data:
        if row.get("db_id") == db_id:
            res.append(row)
    return res


def get_node_key(table):
    name = table.name.lower()
    alias = (table.alias_or_name or table.name).lower()
    return (name, alias)


def get_scope_type(scope):
    # roughly classify what kind of subquery this is
    if scope.scope_type == ScopeType.DERIVED_TABLE:
        return "derived_table"

    select_node = scope.expression
    parent = select_node.parent

    if isinstance(parent, exp.Exists):
        if isinstance(parent.parent, exp.Not):
            return "NOT_EXISTS"
        return "EXISTS"

    if isinstance(parent, exp.Subquery):
        p = parent.parent
        if isinstance(p, exp.In):
            return "IN"
        if isinstance(p, (exp.Any, exp.All)):
            return "ANY_ALL"
        return "scalar"

    return "scalar"


INCLUDE = {"derived_table", "IN", "EXISTS", "NOT_EXISTS", "ANY_ALL"}


def get_depth(scope):
    d = 0
    s = scope
    while s.parent is not None:
        d += 1
        s = s.parent
    return d


def analyze(sql):
    res = {
        "hop_naive": sql.upper().count("JOIN"),
        "hop_breadth": None,
        "hop_diameter": None,
        "has_subquery": False,
        "subquery_types": "none",
        "nesting_depth": 0,
        "needs_manual_review": False,
        "review_reason": "",
        "parse_error": "",
    }

    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception as e:
        res["parse_error"] = str(e)
        res["needs_manual_review"] = True
        res["review_reason"] = "parse_failed"
        return res

    try:
        scopes = traverse_scope(tree)
    except Exception as e:
        res["parse_error"] = str(e)
        res["needs_manual_review"] = True
        res["review_reason"] = "scope_traversal_failed"
        return res

    root = None
    others = []
    for s in scopes:
        if s.scope_type == ScopeType.ROOT:
            root = s
        else:
            others.append(s)

    tables = set(root.tables)
    synth = set()
    excluded = set()
    types_found = set()

    for s in others:
        t = get_scope_type(s)
        types_found.add(t)

        if t == "derived_table":
            # derived table, must catch with outer alias, then join connected
            outer = (s.expression.parent.alias_or_name or "").lower()
            inner = "+".join(sorted({tb.name.lower() for tb in s.tables}))
            synth.add((f"derived({inner})", outer))
        elif t in INCLUDE:
            tables.update(s.tables)
        else:
            for tb in s.tables:
                excluded.add((tb.alias_or_name or tb.name).lower())

    if others:
        res["has_subquery"] = True
        res["subquery_types"] = "+".join(sorted(types_found))
        res["nesting_depth"] = max(get_depth(s) for s in others)

    # build nodes
    nodes = set()
    for t in tables:
        nodes.add(get_node_key(t))
    nodes = nodes | synth

    alias_map = {}
    for name, alias in nodes:
        alias_map[alias] = (name, alias)

    res["hop_breadth"] = max(len(nodes) - 1, 0)

    if len(nodes) <= 1:
        res["hop_diameter"] = 0
        return res

    # build the graph
    g = nx.Graph()
    g.add_nodes_from(nodes)

    for eq in tree.find_all(exp.EQ):
        left = eq.this
        right = eq.expression
        if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
            continue

        la = (left.table or "").lower()
        ra = (right.table or "").lower()

        if la in excluded or ra in excluded:
            continue

        ln = alias_map.get(la)
        rn = alias_map.get(ra)

        if ln and rn and ln != rn:
            g.add_edge(ln, rn)

    if nx.is_connected(g):
        res["hop_diameter"] = nx.diameter(g)
    else:
        comps = list(nx.connected_components(g))
        biggest = max(comps, key=len)
        sub = g.subgraph(biggest)
        if len(biggest) > 1:
            res["hop_diameter"] = nx.diameter(sub)
        else:
            res["hop_diameter"] = 0
        res["needs_manual_review"] = True
        res["review_reason"] = f"disconnected_graph ({len(comps)} components)"

    return res


def main():
    qs = load_questions(DEV_JSON, DB_ID)
    print(f"financial question count: {len(qs)}")

    rows = []
    for q in qs:
        a = analyze(q["SQL"])
        rows.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "difficulty": q["difficulty"],
            "gold_sql": q["SQL"],
            **a,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"save complete -> {OUT_CSV}")

    print("\n- hop_naive -")
    print(df["hop_naive"].value_counts().sort_index())

    print("\n-hop_breadth -")
    print(df["hop_breadth"].value_counts().sort_index())

    print("\n--hop_diameter (final) -")
    print(df["hop_diameter"].value_counts().sort_index())

    diff = df["hop_breadth"] != df["hop_diameter"]
    print(f"\nbreadth and diameter different thing: {diff.sum()} out of whole {len(df)}")
    if diff.sum() > 0:
        print(df.loc[diff, ["question_id", "hop_naive", "hop_breadth", "hop_diameter"]])

    print("\n--subquery types --")
    print(df["subquery_types"].value_counts())

    review = df["needs_manual_review"].sum()
    print(f"\nthing to see manually: {review} things")
    if review > 0:
        print(df.loc[df["needs_manual_review"], ["question_id", "review_reason"]])


if __name__ == "__main__":
    main()
