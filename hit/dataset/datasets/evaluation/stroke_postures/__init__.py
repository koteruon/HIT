import logging

from .stroke_posture_eval import save_stroke_postures_results


def stroke_postures_evaluation(dataset, predictions, output_folder, **_):
    logger = logging.getLogger("hit.inference")
    logger.info("performing stroke postures evaluation.")
    return save_stroke_postures_results(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
