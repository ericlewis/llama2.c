#!/usr/bin/env python3
import argparse
import os
import random
import struct
from typing import Sequence


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"expected positive integer, got {value}")
    return ivalue


def write_floats(file_obj, count: int, values: Sequence[float]) -> None:
    if len(values) != count:
        raise ValueError("value count does not match expected length")
    file_obj.write(struct.pack(f"<{count}f", *values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a dummy LLaMA checkpoint compatible with run.c")
    parser.add_argument("output", help="Path to write the checkpoint")
    parser.add_argument("--dim", type=positive_int, default=64)
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=positive_int, default=128)
    parser.add_argument("--layers", type=positive_int, default=2)
    parser.add_argument("--heads", type=positive_int, default=4)
    parser.add_argument("--kv-heads", dest="kv_heads", type=positive_int, default=4)
    parser.add_argument("--vocab", type=positive_int, default=256)
    parser.add_argument("--seq-len", dest="seq_len", type=positive_int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.dim % args.heads != 0:
        raise SystemExit("dim must be divisible by heads")
    if args.heads % args.kv_heads != 0:
        raise SystemExit("heads must be divisible by kv-heads")

    head_size = args.dim // args.heads
    kv_dim = (args.dim * args.kv_heads) // args.heads

    rng = random.Random(args.seed)

    total_dir = os.path.dirname(args.output)
    if total_dir and not os.path.exists(total_dir):
        os.makedirs(total_dir, exist_ok=True)

    with open(args.output, "wb") as f:
        config = [
            args.dim,
            args.hidden_dim,
            args.layers,
            args.heads,
            args.kv_heads,
            args.vocab,
            args.seq_len,
        ]
        f.write(struct.pack("<7i", *config))

        def rand_values(count: int) -> list[float]:
            return [rng.uniform(-0.01, 0.01) for _ in range(count)]

        def write_block(count: int) -> None:
            write_floats(f, count, rand_values(count))

        write_block(args.vocab * args.dim)
        write_block(args.layers * args.dim)
        write_block(args.layers * args.dim * args.dim)
        write_block(args.layers * args.dim * kv_dim)
        write_block(args.layers * args.dim * kv_dim)
        write_block(args.layers * args.dim * args.dim)
        write_block(args.layers * args.dim)
        write_block(args.layers * args.dim * args.hidden_dim)
        write_block(args.layers * args.hidden_dim * args.dim)
        write_block(args.layers * args.dim * args.hidden_dim)
        write_block(args.dim)
        rope_skip = args.seq_len * head_size // 2
        if rope_skip:
            write_block(rope_skip)
            write_block(rope_skip)

    print(f"Wrote dummy checkpoint to {args.output}")


if __name__ == "__main__":
    main()
