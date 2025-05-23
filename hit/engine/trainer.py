import datetime
import logging
import os
import sys
import time

import torch

from hit.engine.inference import inference
from hit.structures.memory_pool import MemoryPool
from hit.utils.comm import all_gather, reduce_dict, synchronize
from hit.utils.metric_logger import MetricLogger


def do_train(
    cfg,
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    tblogger,
    start_val_period,
    val_period,
    dataset_names_val,
    data_loaders_val,
    distributed,
    mem_active,
):
    logger = logging.getLogger("hit.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    person_pool = arguments["person_pool"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    losses_reduced = torch.tensor(0.0)

    for iteration, (slow_video, fast_video, boxes, objects, keypoints, extras, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        slow_video = slow_video.to(device)
        fast_video = fast_video.to(device)
        boxes = [box.to(device) for box in boxes]
        keypoints = (
            None if all(keypoint is None for keypoint in keypoints) else [keypoint.to(device) for keypoint in keypoints]
        )
        objects = (
            None
            if all(box is None for box in objects)
            else [None if (box is None) else box.to(device) for box in objects]
        )
        mem_extras = {}
        if mem_active:
            movie_ids = [e["movie_id"] for e in extras]
            timestamps = [e["timestamp"] for e in extras]
            mem_extras["person_pool"] = person_pool
            mem_extras["movie_ids"] = movie_ids
            mem_extras["timestamps"] = timestamps
            mem_extras["cur_loss"] = losses_reduced.item()
            mem_extras["video_intex_ts"] = [e["video_intex_ts"] for e in extras if "video_intex_ts" in e]

        loss_dict, weight_dict, metric_dict, pooled_feature = model(
            slow_video, fast_video, boxes, objects, keypoints, mem_extras
        )

        losses = sum([loss_dict[k] * weight_dict[k] for k in loss_dict])

        # reduce losses over all GPUs for logging purposes
        loss_dict["total_loss"] = losses.detach().clone()
        loss_dict_reduced = reduce_dict(loss_dict)
        metric_dict_reduced = reduce_dict(metric_dict)
        meters.update(**loss_dict_reduced, **metric_dict_reduced)
        losses_reduced = loss_dict_reduced.pop("total_loss")

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # update mem pool
        if mem_active:
            person_feature = [ft.to("cpu") for ft in pooled_feature[0]]

            feature_update = MemoryPool()
            for movie_id, timestamp, new_feature in zip(movie_ids, timestamps, person_feature):
                new_feature.add_field("loss_tag", losses_reduced.item())
                feature_update[movie_id, timestamp] = new_feature

            feature_update_list = all_gather(feature_update)
            person_pool.update_list(feature_update_list)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            if tblogger is not None:
                for name, meter in meters.meters.items():
                    tblogger.add_scalar(name, meter.median, iteration)
                tblogger.add_scalar("lr", optimizer.param_groups[0]["lr"], iteration)

        scheduler.step()

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

        if iteration == max_iter:
            arguments.pop("person_pool", None)
            checkpointer.save("model_final", **arguments)

        if (
            dataset_names_val
            and (iteration % val_period == 0 or iteration == start_val_period)
            and iteration >= start_val_period
            and iteration != max_iter
        ):
            # do validation
            val_in_train(
                cfg,
                model,
                dataset_names_val,
                data_loaders_val,
                tblogger,
                iteration,
                start_val_period,
                distributed,
                mem_active,
                checkpointer,
                arguments,
            )
            end = time.time()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / (max_iter)))


def val_in_train(
    cfg,
    model,
    dataset_names_val,
    data_loaders_val,
    tblogger,
    iteration,
    start_val_period,
    distributed,
    mem_active,
    checkpointer,
    arguments,
):
    if distributed:
        model_val = model.module
    else:
        model_val = model
    checkpointer.save("model_{:07d}".format(iteration), **arguments)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", f"{dataset_name}_{iteration}")
            os.makedirs(output_folder, exist_ok=True)
            output_folders[idx] = output_folder
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names_val, data_loaders_val):
        avg_precision = inference(
            model_val,
            data_loader_val,
            dataset_name,
            mem_active,
            output_folder=output_folder,
        )
        synchronize()
    if avg_precision < cfg.SOLVER.TARGET_EVAL_MAP and start_val_period == iteration:
        sys.exit()
    model.train()
