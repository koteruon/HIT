import logging

from .nw_ucla_eval import save_nw_ucla_results


def nw_ucla_evaluation(dataset, predictions, output_folder, **_):
    logger = logging.getLogger("hit.inference")
    logger.info("performing nw_ucla evaluation.")
    return save_nw_ucla_results(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
