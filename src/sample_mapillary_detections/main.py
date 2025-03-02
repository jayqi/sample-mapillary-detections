from pathlib import Path

import geopandas as gpd
from loguru import logger
from pydantic import BaseModel
from shapely import Polygon
from tqdm import tqdm

from sample_mapillary_detections.mapillary import (
    DetectionProportions,
    Image,
    get_detections_for_image,
    get_nearest_image_with_detections,
    get_total_detection_proportions,
    request_images,
)
from sample_mapillary_detections.sampling import halton_sampler


class Sample(BaseModel):
    image: Image
    detection_proportions: DetectionProportions

    def to_row_dict(self) -> dict:
        return self.image.model_dump() | self.detection_proportions


DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SAMPLER = halton_sampler


class MapillaryDetectionsSampler:
    def __init__(
        self, target_geo: Polygon, n_samples: int = DEFAULT_SAMPLE_SIZE, sampler=DEFAULT_SAMPLER
    ):
        self.target_geo = target_geo
        self.n_samples = n_samples
        self.samples: list[Sample] = []
        self.sampler = sampler(target_geo)

    @classmethod
    def from_input_file(
        cls, file_path: str | Path, n_samples: int = DEFAULT_SAMPLE_SIZE, sampler=DEFAULT_SAMPLER
    ):
        gdf = gpd.read_file(file_path)
        return cls(gdf.geometry[0], n_samples=n_samples, sampler=sampler)

    def increase_sample_size_by(self, n_samples: int):
        self.n_samples += n_samples
        logger.info(f"Increased sample size to {self.n_samples}")

    def fetch(self):
        to_fetch = self.n_samples - len(self.samples)
        with tqdm(total=to_fetch) as pbar:
            while len(self.samples) < self.n_samples:
                point = next(self.sampler)
                images_response = request_images(point)
                image = get_nearest_image_with_detections(images_response.json(), point=point)
                if image is None:
                    continue
                detections = get_detections_for_image(image)
                total_proportions = get_total_detection_proportions(detections)
                self.samples.append(Sample(image=image, detection_proportions=total_proportions))
                pbar.update(1)

    def results_to_gdf(self):
        self.fetch()
        return gpd.GeoDataFrame([result.to_row() for result in self.samples])
