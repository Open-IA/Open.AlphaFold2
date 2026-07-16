from pathlib import Path

import numpy as np

from open_alphafold2.visualization import plot_distance_matrix


def test_plot_distance_matrix_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "distance.png"

    result = plot_distance_matrix(np.array([[0.0, 5.0], [5.0, 0.0]]), output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
