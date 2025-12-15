from typing import Iterable, Sequence, Optional
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

def make_colormap(
    hex_colors: Sequence[str],
    positions: Optional[Iterable[float]] = None,
    name: str = "custom",
    N: int = 256,
) -> LinearSegmentedColormap:
    """
    Create a Matplotlib colormap from hex colors with optional positions.

    Parameters
    ----------
    hex_colors : sequence of str
        Hex color strings like '#RRGGBB' (or 'RRGGBB'). At least 2 required.
    positions : iterable of float in [0, 1], optional
        Locations of each color along the gradient. If not provided,
        they are evenly spaced via linspace(0, 1, len(hex_colors)).
        If provided, they do not have to be sorted; they will be sorted
        together with the colors. If the first/last position do not
        hit 0 or 1, they’ll be extended by duplicating the end colors.
    name : str
        Name for the colormap.
    N : int
        Resolution (number of lookup entries) of the colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    # Normalize hex strings and basic validation
    colors = []
    for c in hex_colors:
        if not isinstance(c, str):
            raise TypeError("All colors must be hex strings.")
        colors.append(c if c.startswith("#") else f"#{c}")

    if len(colors) < 2:
        raise ValueError("Provide at least two hex colors.")

    # Build/validate positions
    if positions is None:
        pos = np.linspace(0.0, 1.0, len(colors), dtype=float)
    else:
        pos = np.asarray(list(positions), dtype=float)
        if pos.size != len(colors):
            raise ValueError("`positions` must have the same length as `hex_colors`.")
        if np.any((pos < 0.0) | (pos > 1.0)):
            raise ValueError("All positions must be within [0, 1].")

        # Sort positions and carry colors along
        order = np.argsort(pos)
        pos = pos[order]
        colors = [colors[i] for i in order]

        # Ensure coverage of [0, 1] by extending ends if needed
        if pos[0] > 0.0:
            pos = np.r_[0.0, pos]
            colors = [colors[0]] + colors
        if pos[-1] < 1.0:
            pos = np.r_[pos, 1.0]
            colors = colors + [colors[-1]]

    # Pair positions with RGBA
    segment = list(zip(pos.tolist(), (to_rgba(c) for c in colors)))

    # Create the colormap
    return LinearSegmentedColormap.from_list(name, segment, N=N)
