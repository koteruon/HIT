import logging
from .table_tennis_eval import save_results


def table_tennis_evaluation(dataset, predictions, output_folder, **_):
    logger = logging.getLogger("hit.inference")
    logger.info("performing table_tennis evaluation.")
    return save_results(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
