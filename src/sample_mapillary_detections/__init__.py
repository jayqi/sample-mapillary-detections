import base64
from datetime import datetime
from itertools import islice
import json
import os
from typing import Annotated, Iterator

import geopandas as gpd
from geopy.distance import geodesic
import httpx
from loguru import logger
import mapbox_vector_tile
from pydantic import BaseModel, BeforeValidator, ConfigDict, field_serializer
import shapely
from shapely import Point, Polygon


def halton_sequence(b):
    """Generator function for Halton sequence."""
    n, d = 0, 1
    while True:
        x = d - n
        if x == 1:
            n = 1
            d *= b
        else:
            y = d // b
            while x <= y:
                y //= b
            n = (b + 1) * y - x
        yield n / d


def halton_2d(b1=2, b2=3):
    """Generate 2D Halton sequence."""
    seq1, seq2 = halton_sequence(b1), halton_sequence(b2)
    while True:
        yield (next(seq1), next(seq2))


def geo_halton_2d(polygon: Polygon) -> Iterator[Point]:
    min_x, min_y, max_x, max_y = polygon.bounds
    delta_x = max_x - min_x
    delta_y = max_y - min_y

    seq = halton_2d()
    while True:
        x, y = next(seq)
        x = min_x + x * delta_x
        y = min_y + y * delta_y
        if polygon.contains(Point(x, y)):
            yield Point(x, y)


class Bbox(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @classmethod
    def from_point(cls, point: Point, extent_meters: float = 100.0):
        """Create a bounding box centered on a point with a given extent in meters."""
        lat_lon = (point.y, point.x)
        north = geodesic(meters=extent_meters).destination(lat_lon, 0)
        east = geodesic(meters=extent_meters).destination(lat_lon, 90)
        south = geodesic(meters=extent_meters).destination(lat_lon, 180)
        west = geodesic(meters=extent_meters).destination(lat_lon, 270)
        return cls(
            min_lon=west.longitude,
            min_lat=south.latitude,
            max_lon=east.longitude,
            max_lat=north.latitude,
        )


class ImagesParams(BaseModel):
    """Parameters for the Mapillary images API."""

    bbox: Bbox
    is_pano: bool = True
    limit: int = 20
    fields: list[str] = [
        "id",
        "geometry",
        "detections",
    ]
    start_captured_at: datetime = None
    end_captured_at: datetime = None

    @field_serializer("bbox")
    def serialize_bbox(self, bbox, _info):
        return ",".join(str(val) for val in bbox.model_dump().values())

    @field_serializer("fields")
    def serialize_fields(self, fields, _info):
        return ",".join(field for field in fields)


def request_images(point: Point) -> httpx.Response:
    params = ImagesParams(bbox=Bbox.from_point(point))
    params_dict = params.model_dump(exclude_none=True)
    access_token = os.getenv("MAPILLARY_CLIENT_TOKEN")
    headers = {"Authorization": f"OAuth {access_token}"}
    resp = httpx.get(
        "https://graph.mapillary.com/images",
        params=params_dict,
        headers=headers,
    )
    return resp


class Image(BaseModel):
    id: str
    geometry: Point

    model_config = ConfigDict(arbitrary_types_allowed=True)


def get_first_image_with_detections(images_response: dict):
    for image in images_response["data"]:
        if image.get("detections", {}).get("data", []):
            return Image(
                id=image["id"], geometry=shapely.from_geojson(json.dumps(image["geometry"]))
            )
    return None


def request_detections(image_id: str) -> httpx.Response:
    url = f"https://graph.mapillary.com/{image_id}/detections"
    params_dict = {"fields": "value,geometry"}
    access_token = os.getenv("MAPILLARY_CLIENT_TOKEN")
    headers = {"Authorization": f"OAuth {access_token}"}
    resp = httpx.get(url, params=params_dict, headers=headers)
    return resp


class DetectionGeometry(BaseModel):
    extent: int
    features: list[Polygon]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def area_proportion(self):
        return sum(feature.area for feature in self.features) / self.extent**2


def geometry_string_to_detection_geometry(geometry_string: str) -> DetectionGeometry:
    """Decodes a base64-encoded Mapbox Vector Tile geometry string returned by the Mapillary API
    into a DetectionGeometry object.

    https://www.mapillary.com/developer/api-documentation#detection
    """
    decoded_data = base64.decodebytes(geometry_string.encode("utf-8"))
    data_dict = mapbox_vector_tile.decode(decoded_data)
    extent = data_dict["mpy-or"]["extent"]
    features = [
        shapely.from_geojson(json.dumps(feature["geometry"]))
        for feature in data_dict["mpy-or"]["features"]
    ]
    return DetectionGeometry(extent=extent, features=features)


class Detection(BaseModel):
    value: str
    geometry: Annotated[DetectionGeometry, BeforeValidator(geometry_string_to_detection_geometry)]


def get_detections_for_image(image: Image) -> list[Detection]:
    detections_response = request_detections(image.id)
    detections = [
        Detection.model_validate(record)
        for record in detections_response.json()["data"]
        if record["value"] in DETECTION_TYPES
    ]
    return detections


DETECTION_TYPES = (
    "nature--terrain",
    "nature--vegetation",
    "nature--water",
    "nature--sky",
    "construction--structure--building",
)

from typing import TypedDict

DetectionProportions = TypedDict(
    "DetectionProportions", {detection_type: float for detection_type in DETECTION_TYPES}
)


def get_total_detection_proportions(detections: list[Detection]) -> DetectionProportions:
    total_proportions: DetectionProportions = {
        detection_type: 0.0 for detection_type in DETECTION_TYPES
    }
    for detection in detections:
        total_proportions[detection.value] += detection.geometry.area_proportion
    return total_proportions


def main(polygon: Polygon, n_points: int = 10):
    results = []
    for point in islice(geo_halton_2d(polygon), n_points):
        images_response = request_images(point)
        image = get_first_image_with_detections(images_response.json())
        if image is None:
            continue
        detections = get_detections_for_image(image)
        total_proportions = get_total_detection_proportions(detections)
        results.append(image.model_dump() | total_proportions)

    return gpd.GeoDataFrame(results)
