from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult


class Predictor:
    def __init__(self, config, verbose: bool = False):
        self.detector = Detector(config.detector_weights, config.device, verbose=verbose)
        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=True,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw

    def recognize(self, image: np.ndarray) -> Tuple[PersonAndFaceResult, Optional[np.ndarray], defaultdict]:
        detected_objects: PersonAndFaceResult = self.detector.predict(image)
        cur_result = self.age_gender_model.predict(image, detected_objects)
        
        out_im = None
        if self.draw:
            # plot results on image
            out_im = detected_objects.plot()

        return detected_objects, out_im, cur_result
