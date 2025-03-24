from hit.dataset import datasets

# from .jhmdb import jhmdb_evaluation
from .stroke_postures import stroke_postures_evaluation

# from .table_tennis import table_tennis_evaluation

# from .ava import ava_evaluation


def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs)
    if isinstance(dataset, datasets.DatasetEngine):
        # return ava_evaluation(**args)
        # return table_tennis_evaluation(**args)
        # return jhmdb_evaluation(**args)
        return stroke_postures_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
