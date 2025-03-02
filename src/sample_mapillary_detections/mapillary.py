import base64
from datetime import datetime
import json
import os
from typing import Annotated, TypedDict

from geopy.distance import geodesic
import httpx
import mapbox_vector_tile
from pydantic import BaseModel, BeforeValidator, ConfigDict, field_serializer
import shapely
from shapely import Point, Polygon
import stamina

timeout_config = httpx.Timeout(10.0, read=30.0)


def retry_predicate(exception: Exception) -> bool:
    if isinstance(exception, httpx.ReadTimeout):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        return 500 <= exception.response.status_code < 600
    return False


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
    limit: int = 50
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


@stamina.retry(on=retry_predicate, attempts=5, wait_exp_base=10)
def request_images(point: Point) -> httpx.Response:
    params = ImagesParams(bbox=Bbox.from_point(point))
    params_dict = params.model_dump(exclude_none=True)
    access_token = os.getenv("MAPILLARY_CLIENT_TOKEN")
    headers = {"Authorization": f"OAuth {access_token}"}
    resp = httpx.get(
        "https://graph.mapillary.com/images",
        params=params_dict,
        headers=headers,
        timeout=timeout_config,
    )
    resp.raise_for_status()
    return resp


class Image(BaseModel):
    id: str
    geometry: Point

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_image_data(cls, image_data: dict):
        return cls(
            id=image_data["id"],
            geometry=shapely.from_geojson(json.dumps(image_data["geometry"])),
        )


def get_first_image_with_detections(images_data: dict):
    for image in images_data["data"]:
        if image.get("detections", {}).get("data", []):
            return Image(
                id=image["id"], geometry=shapely.from_geojson(json.dumps(image["geometry"]))
            )
    return None


def get_nearest_image_with_detections(images_data: dict, point: Point) -> Image | None:
    nearest_image: Image | None = None
    nearest_image_dist: float | None = None
    for image_data in images_data["data"]:
        if image_data.get("detections", {}).get("data", []):
            image = Image.from_image_data(image_data)
            image_dist = point.distance(image.geometry)
            if nearest_image_dist is None or image_dist < nearest_image_dist:
                nearest_image = image
                nearest_image_dist = image_dist
    return nearest_image


@stamina.retry(on=retry_predicate, attempts=5, wait_exp_base=10)
def request_detections(image_id: str) -> httpx.Response:
    url = f"https://graph.mapillary.com/{image_id}/detections"
    params_dict = {"fields": "value,geometry"}
    access_token = os.getenv("MAPILLARY_CLIENT_TOKEN")
    headers = {"Authorization": f"OAuth {access_token}"}
    resp = httpx.get(url, params=params_dict, headers=headers, timeout=timeout_config)
    resp.raise_for_status()
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
