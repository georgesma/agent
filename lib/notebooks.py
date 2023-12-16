import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np


def plot_groups_metrics(groups_metrics, metrics_name, split_name="test"):
    nb_groups = len(groups_metrics)
    nb_metrics = len(metrics_name)

    groups_name = list(groups_metrics.keys())
    groups_name.sort()

    plt.figure(figsize=(nb_metrics * 3 + 3, nb_groups * 3), dpi=100)
    for i_group, group_name in enumerate(groups_name):
        group_metrics = groups_metrics[group_name]

        for i_metric, metric_name in enumerate(metrics_name):
            ax = plt.subplot(nb_groups, nb_metrics, 1 + nb_metrics * i_group + i_metric)
            if i_metric == 0:
                ax.text(
                    -0.3,
                    0.5,
                    group_name,
                    verticalalignment="center",
                    horizontalalignment="right",
                    transform=ax.transAxes,
                )
            for metrics in group_metrics.values():
                if metric_name not in metrics[split_name]:
                    continue
                ax.set_title(metric_name)
                ax.plot(metrics[split_name][metric_name])
                last_epoch = len(metrics[split_name][metric_name]) - 1
                last_value = metrics[split_name][metric_name][-1]
                ax.scatter(last_epoch, last_value, zorder=10)
    plt.tight_layout()
    # plt.subplots_adjust(hspace=.25, wspace=.25)
    plt.show()


def show_ema(ema, reference=None, dataset=None):
    nb_frames = len(ema)
    ema_x, ema_y = ema[:, 0::2], ema[:, 1::2]
    if dataset is not None:
        xlim = (dataset.ema_limits["xmin"] * 0.95, dataset.ema_limits["xmax"] * 1.05)
        ylim = (dataset.ema_limits["ymin"] * 0.95, dataset.ema_limits["ymax"] * 1.05)
    else:
        xlim = (ema_x.min() * 0.95, ema_x.max() * 1.05)
        ylim = (ema_y.min() * 0.95, ema_y.max() * 1.05)

    def show_ema_frame(
        i_frame=0, trail_len=20, show_reference=True, show_reference_trail=False
    ):
        trail_opacity = np.linspace(0, 0.5, trail_len)
        trail_len = min(i_frame, trail_len - 1)
        trail_opacity = trail_opacity[-trail_len:]
        trail_start = i_frame - trail_len

        plt.figure()
        ax = plt.subplot()
        if dataset is not None and dataset.has_palate:
            ax.plot(dataset.palate[:, 0], dataset.palate[:, 1])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.scatter(ema_x[i_frame], ema_y[i_frame], c="tab:blue")
        if trail_len > 0:
            ax.scatter(
                ema_x[trail_start:i_frame],
                ema_y[trail_start:i_frame],
                alpha=trail_opacity,
                c="tab:blue",
                s=10,
            )
        if show_reference and reference is not None:
            ax.scatter(reference[i_frame, 0::2], reference[i_frame, 1::2], c="tab:red")
            if show_reference_trail:
                ax.scatter(
                    reference[trail_start:i_frame, 0::2],
                    reference[trail_start:i_frame, 1::2],
                    alpha=trail_opacity,
                    c="tab:red",
                    s=10,
                )
        plt.show()

    ipw.interact(
        show_ema_frame,
        i_frame=(0, nb_frames - 1),
        trail_len=(0, 50, 1),
        show_reference=True,
        show_reference_trail=False,
    )


def show_occlusions_metrics(phones_metrics, palate):
    distances = ["lips", "tongue_tip", "tongue_mid"]

    def show_phone_metrics(phone):
        consonant_metrics = phones_metrics[phone]

        plt.figure(dpi=120)

        for i_distance, distance in enumerate(distances):
            distance_ema = np.array(consonant_metrics["min_%s_ema" % distance])
            ax = plt.subplot(3, 2, 1 + 2 * i_distance, aspect="equal")
            ax.plot(palate[:, 0], palate[:, 1])
            ax.scatter(distance_ema[:, 0::2], distance_ema[:, 1::2], s=1)

            ax = plt.subplot(3, 2, 2 + 2 * i_distance)
            ax.hist(consonant_metrics["min_%s_distance" % distance])

        plt.tight_layout()
        plt.show()

    phones = phones_metrics.keys()
    ipw.interact(show_phone_metrics, phone=phones)
