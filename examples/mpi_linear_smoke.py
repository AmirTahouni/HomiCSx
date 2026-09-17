"""Two-or-more-rank MPI smoke test for the supported linear workflow."""

from __future__ import annotations

import json

import numpy as np
from mpi4py import MPI

from linear_periodic_2d import run_example


def main() -> int:
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        raise RuntimeError("Run this smoke test with mpiexec -n 2 or more.")

    summary = run_example()
    gathered = comm.gather(summary, root=0)
    if comm.rank == 0:
        reference = gathered[0]
        for rank_summary in gathered[1:]:
            if rank_summary["shape"] != reference["shape"]:
                raise AssertionError("MPI ranks returned inconsistent tensor shapes.")
            if not np.isclose(
                rank_summary["trace"], reference["trace"], rtol=1e-12, atol=1e-12
            ):
                raise AssertionError(
                    f"MPI ranks returned inconsistent stiffness: {gathered!r}"
                )
        print(json.dumps({"mpi_ranks": comm.size, **reference}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
