"""Generates data for the group sequence prediction task."""

import os
import random
from functools import reduce
from itertools import product, permutations
from pathlib import Path

import fire
import polars as pl
import pyrootutils
from abstract_algebra.finite_algebras import (
    FiniteAlgebra,
    generate_cyclic_group,
    generate_symmetric_group,
    make_finite_algebra,
)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


def group_reduce(lhs: str | int, rhs: int, G) -> int:  # noqa: N803
    """Reduce a sequence of group elements to a single element."""
    if isinstance(lhs, str):
        prod = G.op(lhs, G.elements[rhs])
    else:
        prod = G.op(G.elements[lhs], G.elements[rhs])

    return G.elements.index(prod)


def generate_heisenberg_group(p: int, name=None):
    """
    Finite Heisenberg group H_3(Z_p).
    Elements are triples (a,b,c) with group law:
      (a,b,c)*(a',b',c') = (a+a', b+b', c+c'+a*b') mod p
    """
    if p <= 1:
        raise ValueError("p must be >= 2")

    # element names
    elems = [f"({a},{b},{c})"
             for a in range(p)
             for b in range(p)
             for c in range(p)]

    def mul(x, y):
        a,b,c = x
        a2,b2,c2 = y
        return ((a+a2) % p,
                (b+b2) % p,
                (c+c2 + a*b2) % p)

    sym2elt = {}
    elt2sym = {}

    for a in range(p):
        for b in range(p):
            for c in range(p):
                s = f"({a},{b},{c})"
                t = (a,b,c)
                sym2elt[s] = t
                elt2sym[t] = s

    table = []
    for e1 in elems:
        row = []
        for e2 in elems:
            prod = mul(sym2elt[e1], sym2elt[e2])
            row.append(elt2sym[prod])
        table.append(row)

    group_name = name or f"H3_Z{p}"
    desc = f"Heisenberg group H_3(Z_{p}) of order {p**3}"
    return make_finite_algebra(group_name, desc, elems, table)


def generate_dihedral_group_from_order(order: int, name: str | None = None) -> FiniteAlgebra:
    """
    Create the dihedral group of given *group order*.
    Conventions:
      - D8 means the dihedral group of order 8 (symmetries of a square),
        i.e. <r,s | r^4=e, s^2=e, srs=r^{-1}>.
      - In general, dihedral group of order 2n is returned for `order=2n`.
    """
    if order <= 0 or order % 2 != 0:
        raise ValueError(f"Dihedral group order must be a positive even integer, got {order}.")

    n = order // 2  # number of rotations
    # Elements are r^k and s r^k for k=0..n-1
    # We'll name them: r0..r(n-1), s0..s(n-1)
    elements = [f"r{k}" for k in range(n)] + [f"s{k}" for k in range(n)]

    def mul(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
        # Represent r^k as (k,0) and s r^k as (k,1)
        (k, flip_a), (l, flip_b) = a, b
        # (k,flip) * (l,flip2) = (k + (-1)^flip * l mod n, flip xor flip2)
        if flip_a == 0:
            new_k = (k + l) % n
        else:
            new_k = (k - l) % n
        return (new_k, flip_a ^ flip_b)

    # Precompute symbol -> (k,flip) and back
    sym2pair: dict[str, tuple[int, int]] = {}
    pair2sym: dict[tuple[int, int], str] = {}

    for k in range(n):
        sym2pair[f"r{k}"] = (k, 0)
        pair2sym[(k, 0)] = f"r{k}"
        sym2pair[f"s{k}"] = (k, 1)
        pair2sym[(k, 1)] = f"s{k}"

    # Cayley table as list[list[str]] with row/col indexed by `elements`
    table: list[list[str]] = []
    for e1 in elements:
        row = []
        for e2 in elements:
            p = mul(sym2pair[e1], sym2pair[e2])
            row.append(pair2sym[p])
        table.append(row)

    alg_name = name or f"D{order}"
    description = f"Dihedral group of order {order} (2*{n})"
    return make_finite_algebra(alg_name, description, elements, table)


def generate_group(g: (str, int)) -> FiniteAlgebra:
    """Generate an group from a string identifier."""
    if g[0] == "S":
        return generate_symmetric_group(g[1])
    elif g[0] == "Z":
        return generate_cyclic_group(g[1])
    elif g[0] == "A":
        s_n = generate_symmetric_group(g[1])
        a_n = s_n.commutator_subalgebra()
        a_n.name = f"A{g[1]}"
        return a_n
    elif g[0] == "D":
        return generate_dihedral_group_from_order(g[1])
    elif g[0] == "H":
        # e.g., H3_2 or H3_3 but hardcoded to H3_2
        p = 2
        return generate_heisenberg_group(p)
    else:
        raise ValueError("Group must be one of S, Z, D, H, or A")


def main(
    group: str,
    k: int | list[int] = 10,
    samples: int | None = None,
    data_dir: str | Path = PROJECT_ROOT / "data",
    seed: int = random.randint(0, 1_000_000),
    overwrite: bool = False,
):
    """Generate data for the group sequence prediction task."""
    data_path = data_dir / f"{group}={k}.csv"
    if data_path.exists() and not overwrite:
        print(
            f"Data already exists at {data_path}. Use `--overwrite` to regenerate file."
        )
        return

    random.seed(seed)
    print(f"Using seed {seed}")

    if group == "S5_only_swaps" or group == "S5_only_swaps_hard":
        group_list = [generate_group(("S", 5))]
        group_prod = reduce(lambda x, y: x * y, group_list)
        
        # Get indices of two element swaps and identity
        ident = tuple(range(1, 6))
        perms = list(permutations(ident))
        allowed_indices = [i for i, perm in enumerate(perms) if sum(x != y for x, y in zip(ident, perm)) <= 2]

        num_elements = len(allowed_indices)
        num_unique_sequences = num_elements**k
    elif 'limit_to' in group:
        g = group[0]
        n = int(group[1])
        limit = int(group[-1])
        assert 1 < limit < n, "Choose a suitable limit"
        assert g == "S", "Only works with permutation groups"
        
        group_list = [generate_group((g, n))]
        group_prod = reduce(lambda x, y: x * y, group_list)
        
        # Get indices of two element swaps and identity
        ident = tuple(range(1, n+1))
        perms = list(permutations(ident))
        allowed_indices = [i for i, perm in enumerate(perms) if sum(x != y for x, y in zip(ident, perm)) <= limit]
        
        num_elements = len(allowed_indices)
        num_unique_sequences = num_elements**k

    else:
        if "tokens" in group:
            g = group.split("_")[0]
            group_ids = [(g[0], int(g[1:]))]
        else:
            group_ids = [(g[0], int(g[1:])) for g in group.split("_x_")]

        group_list = [generate_group(g) for g in group_ids]
    
        group_prod = reduce(lambda x, y: x * y, group_list)
        num_elements = len(group_prod.elements)
        num_unique_sequences = num_elements**k
        allowed_indices = range(num_elements)

    print(f"allowed indices = {allowed_indices}")
    print(f"num_elements = {num_elements}") 

    if samples is None:
        print(
            f"Generating all {num_elements} ^ {k} = " f"{num_elements ** k} sequences."
        )
        print("Output data will not be shuffled.")

        sequences = product(allowed_indices, repeat=k)

    else:
        if samples > num_unique_sequences:
            print(
                f"Warning: {samples} > {num_unique_sequences}. I will only"
                f" generate {num_unique_sequences} examples."
            )
            samples = num_unique_sequences
        print(f"Randomly sampling {samples} sequences.")
        sequences = set()
        while len(sequences) < samples:
            sequences.add(tuple(random.choices(allowed_indices, k=k)))
        sequences = list(sequences)

    examples = []
    for seq in sequences:        
        if group.endswith('hard'):
            inputs = []
            acc = 0
            for i, x in enumerate(seq):
                acc = group_reduce(lhs=acc, rhs=x, G=group_prod)
                if i % 4 == 3:
                    inputs.extend((acc, 0, 0, 0))
                    acc = 0
                elif i == len(seq)-1:
                    inputs.extend((acc, *[0 for _ in range(i % 4)]))
            
            acc = 0
            outputs = [acc := group_reduce(lhs=acc, rhs=x, G=group_prod) for x in inputs]
            
            # shift outputs to make the model learn better
            outputs = [*(0,0,0), *(outputs[:-3])]

        elif "tokens" in group:
            new_length = len(seq)
            n_tokens = int(group.split('_')[1])
            seq = seq[:(new_length//n_tokens)+1]
            acc = 0
            out = [acc := group_reduce(lhs=acc, rhs=x, G=group_prod) for x in seq]
            inputs = []
            outputs = []
            for x, y in zip(seq, out):
                inputs.append(x)
                outputs.append(y)
                if len(inputs) >= new_length:
                    break
                for i in range(n_tokens-1):
                    s_token = None
                    if "s_token" in group:
                        s_token = num_elements
                        inputs.append(s_token)
                        if not "only_input" in group:
                            outputs.append(s_token)
                        else:
                            outputs.append(y)
                    else:
                        inputs.append(x)
                        outputs.append(y)
                    if len(inputs) >= new_length:
                        break
                if len(inputs) >= new_length:
                    break
            # shift supervision
            fill = [0 for _ in range(n_tokens-1)]
            outputs = [*fill, *outputs[:-(n_tokens-1)]]
        else:
            inputs = seq
            
            acc = 0
            outputs = [acc := group_reduce(lhs=acc, rhs=x, G=group_prod) for x in inputs]
  
        assert len(inputs) == len(outputs), f"{len(inputs)}, {len(outputs)}, {len(seq)}"

        examples.append(
            {
                "seed": seed,
                "input": " ".join(map(str, inputs)),
                "target": " ".join(map(str, outputs)),
            }
        )
    if "tokens" in group:
        print(f"n_tokens per symbol: {n_tokens}, fill:{fill}, s_token:{s_token}")
    
    ex_df = pl.from_dicts(examples)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print(f"Writing data to `{data_path}`")
    ex_df.write_csv(data_path)


if __name__ == "__main__":
    fire.Fire(main)
