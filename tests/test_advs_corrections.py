"""
Validation test for ADVS ToT and ToA phase corrections.

Verifies that --hits_tot_correct_file and --hits_toa_phase_correction are
actually applied when parsing .advs files (they were previously no-ops).
"""

import os
import sys
import struct
import tempfile
from pathlib import Path

import numpy as np
import h5py

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.constants import dt_hit
from tpx3format.read_advs import parse_advs_data_package, parse_advs_data_packages
from tpx3format.read import read_tot_correction, apply_tot_correction


def create_synthetic_advs_file(path: str, n_packets: int = 10) -> None:
    """Create a minimal synthetic ADVS file (16 bytes per packet)."""
    with open(path, 'wb') as f:
        for i in range(n_packets):
            # First packet at x=10, y=20 for ToT correction test; others vary
            if i == 0:
                addr = 10 + 20 * 256  # x=10, y=20
                tot = 50
            else:
                addr = (i % 256) + ((i * 37) % 256) * 256
                tot = 50 + (i % 100)
            toa = 1000 + i * 1000
            f.write(struct.pack('<IIQ', addr, tot, toa))


def create_minimal_tot_correction_file(path: str) -> None:
    """Create a minimal ToT correction HDF5 with non-zero corrections."""
    # Shape: (1024, 256, 256, 1) for single-chip
    shape = (1024, 256, 256, 1)
    # Non-zero correction: add +5 for ToT=50, x=10, y=20
    data = np.zeros(shape, dtype=np.int16)
    data[50, 20, 10, 0] = 5  # (ToT, y, x, chip_id)
    with h5py.File(path, 'w') as f:
        f.create_dataset('tot_correction', data=data)
        f['tot_correction'].attrs['creation_date'] = '2025-01-01'


def test_tot_correction_applied():
    """Verify ToT correction changes hit ToT values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        advs_path = os.path.join(tmpdir, 'test.advs')
        tot_path = os.path.join(tmpdir, 'tot_correct.h5')

        create_synthetic_advs_file(advs_path, n_packets=5)
        create_minimal_tot_correction_file(tot_path)

        tot_correction = read_tot_correction(tot_path)
        assert tot_correction is not None

        # Parse WITHOUT correction
        class SettingsNoCorr:
            hits_tot_threshold = 0
            hits_toa_phase_correction = 0
            hits_toa_lut_file = "0"

        with open(advs_path, 'rb') as f:
            pos = [0, 5 * 16, 0, 0]  # 5 packets
            hits_no_corr = list(parse_advs_data_package(
                f, pos, None, 0, None, 0
            ))

        # Parse WITH correction
        with open(advs_path, 'rb') as f:
            hits_with_corr = list(parse_advs_data_package(
                f, pos, tot_correction, 0, None, 0
            ))

        # Filter Nones and compare
        hits_no_corr = [h for h in hits_no_corr if h is not None]
        hits_with_corr = [h for h in hits_with_corr if h is not None]

        assert len(hits_no_corr) > 0 and len(hits_with_corr) > 0

        # At least one hit should have different ToT (the one at x=10, y=20, tot=50)
        tot_differ = False
        for (c0, x0, y0, tot0, toa0), (c1, x1, y1, tot1, toa1) in zip(hits_no_corr, hits_with_corr):
            if tot0 != tot1:
                tot_differ = True
                break
        assert tot_differ, "ToT correction should change at least one hit's ToT"


def test_toa_phase_correction_applied():
    """Verify ToA phase correction changes hit ToA values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        advs_path = os.path.join(tmpdir, 'test.advs')
        create_synthetic_advs_file(advs_path, n_packets=5)

        with open(advs_path, 'rb') as f:
            pos = [0, 5 * 16, 0, 0]
            hits_no_phase = list(parse_advs_data_package(
                f, pos, None, 0, None, 0
            ))

        with open(advs_path, 'rb') as f:
            hits_phase3 = list(parse_advs_data_package(
                f, pos, None, 0, None, 3  # Pll94
            ))

        hits_no_phase = [h for h in hits_no_phase if h is not None]
        hits_phase3 = [h for h in hits_phase3 if h is not None]

        assert len(hits_no_phase) > 0 and len(hits_phase3) > 0

        # Phase correction 3 modifies columns 196,197,200,201,204,205 and x%4 in {2,3}
        toa_differ = any(
            toa0 != toa1
            for (_, _, _, _, toa0), (_, _, _, _, toa1) in zip(hits_no_phase, hits_phase3)
        )
        assert toa_differ, "ToA phase correction should change at least one hit's ToA"


if __name__ == '__main__':
    test_tot_correction_applied()
    print("✓ ToT correction test passed")
    test_toa_phase_correction_applied()
    print("✓ ToA phase correction test passed")
    print("\nAll ADVS correction tests passed.")
