# sample-mapillary-detections

A proof-of-concept Python package for analyzing panoramic street view imagery from Mapillary. Given an input geographical region as a polygon, it will:

- Sample locations in the region with a [low-discrepancy](https://en.wikipedia.org/wiki/Low-discrepancy_sequence) algorithm
- Query Mapillary for images near each sampled location and select the nearest image that has segmentation results
- Calculate the total proportion of view area for a select set of segmentation classes

See [`notebooks/demo.ipynb`](./notebooks/demo.ipynb) for an demonstration using [sample input data](./data/BeverlyHills.geojson).

This project was developed in association with [Civic Tech DC](https://www.civictechdc.org/).

## Installation

1. Clone this repository and set it as your working directory.

    ```bash
    git clone https://github.com/jayqi/sample-mapillary-detections.git
    cd sample-mapillary-detections
    ```

2. Install dependencies. Using [uv](https://docs.astral.sh/uv/) is recommended to reproduce the exact environment from the provided lockfile.

    ```bash
    uv sync
    ```

    Alternatively, you can install the package in any typical way, such as with Pip. **This approach is less reliable as it will not use locked dependency versions.** Using a virtual environment is recommended. This project was developed in Python 3.12.

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -e .
    ```
