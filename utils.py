from typing import List

from mne.channels import DigMontage


def save_montage(montage: DigMontage, montage_path: str) -> None:
    """Saves montage in matlab-compatible XYZ format

    Args:
        montage: DigMontage object extracted from the sample
        montage_path: Exact path were to save the montage file
    """
    rows = []
    for ch, coords in montage.get_positions()['ch_pos'].items():
        rows.append([ch, coords[0], coords[1], coords[2]])
    res = ''
    for row in rows:
        res += f'{row[0]} {row[1]:.4f} {row[2]:.4f} {row[3]:.4f}\n'
    with open(montage_path, 'w') as f:
        f.write(res)
