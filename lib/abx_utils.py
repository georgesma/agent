import random
import numpy as np
from librosa.sequence import dtw


def get_phones_indexes(items_lab, phones, contexts):
    phones_indexes = {phone: [] for phone in phones}

    for item_name, item_lab in items_lab.items():
        if len(item_lab) < 3:
            continue
        lab_len = len(item_lab)
        for i in range(lab_len - 2):
            if (
                item_lab[i]["name"] in contexts
                and item_lab[i + 1]["name"] in phones
                and item_lab[i + 2]["name"] in contexts
            ):
                phone_label = item_lab[i + 1]
                phone_name = phone_label["name"]
                phone_index = slice(phone_label["start"], phone_label["end"])
                phones_indexes[phone_name].append((item_name, phone_index))

    return phones_indexes


def get_datasets_phones_indexes(datasets_items_lab, phones, contexts):
    phones_indexes = {phone: [] for phone in phones}

    for dataset_name, items_lab in datasets_items_lab.items():
        for item_name, item_lab in items_lab.items():
            if len(item_lab) < 3:
                continue
            lab_len = len(item_lab)
            for i in range(lab_len - 2):
                if (
                    item_lab[i]["name"] in contexts
                    and item_lab[i + 1]["name"] in phones
                    and item_lab[i + 2]["name"] in contexts
                ):
                    phone_label = item_lab[i + 1]
                    phone_name = phone_label["name"]
                    phone_index = slice(phone_label["start"], phone_label["end"])
                    phones_indexes[phone_name].append(
                        (dataset_name, item_name, phone_index)
                    )

    return phones_indexes


def get_phone_features(datasets_features, features_type, phone_index):
    return datasets_features[phone_index[0]][features_type][phone_index[1]][
        phone_index[2]
    ]


def get_abx_matrix(
    phones, phones_indexes, datasets_features, distance, samples_per_pair
):
    nb_phones = len(phones)
    matrix_nb_success = np.zeros((nb_phones, nb_phones), dtype="int")
    matrix_nb_test = np.zeros((nb_phones, nb_phones), dtype="int")

    for i_ax_phone, ax_phone in enumerate(phones):
        for i_b_phone, b_phone in enumerate(phones):
            if ax_phone == b_phone:
                continue
            ax_indexes = phones_indexes[ax_phone]
            b_indexes = phones_indexes[b_phone]

            nb_ax_indexes = len(ax_indexes)
            nb_b_indexes = len(b_indexes)
            if nb_ax_indexes < 2 or nb_b_indexes < 1:
                continue

            for _ in range(samples_per_pair):
                a_index, x_index = random.sample(ax_indexes, 2)
                b_index = random.choice(b_indexes)

                if conduct_abx_test(
                    distance, datasets_features, a_index, b_index, x_index
                ):
                    matrix_nb_success[i_ax_phone, i_b_phone] += 1
                matrix_nb_test[i_ax_phone, i_b_phone] += 1

    return matrix_nb_success, matrix_nb_test


def get_features_distance(x_features, y_features, distance_metric):
    cost_matrix, warping_path = dtw(x_features.T, y_features.T, metric=distance_metric)
    final_cost = cost_matrix[warping_path[0, 0], warping_path[0, 1]]
    path_len = len(warping_path)
    mean_cost = final_cost / path_len
    return mean_cost


def conduct_abx_test(distance, datasets_features, a_index, b_index, x_index):
    ax_distance = 0
    bx_distance = 0

    for features_type, feature_distance in distance.items():
        a_features = get_phone_features(datasets_features, features_type, a_index)
        b_features = get_phone_features(datasets_features, features_type, b_index)
        x_features = get_phone_features(datasets_features, features_type, x_index)

        ax_features_distance = get_features_distance(
            a_features, x_features, feature_distance["metric"]
        )
        bx_features_distance = get_features_distance(
            b_features, x_features, feature_distance["metric"]
        )

        ax_distance += ax_features_distance * feature_distance["weight"]
        bx_distance += bx_features_distance * feature_distance["weight"]

    return ax_distance < bx_distance


def get_global_score(abx_matrix):
    return abx_matrix[0].sum() / abx_matrix[1].sum() * 100


def get_groups_score(phones, abx_matrix, groups):
    groups_score = {}

    for group_name, subgroups in groups.items():
        subgroups_score = []
        for subgroup in subgroups:
            subgroup_nb_success = 0
            subgroup_nb_test = 0

            subgroup_indexes = [phones.index(phone) for phone in subgroup]
            for i in subgroup_indexes:
                for j in subgroup_indexes:
                    if i == j:
                        continue
                    subgroup_nb_success += abx_matrix[0][i, j]
                    subgroup_nb_test += abx_matrix[1][i, j]

            subgroup_score = subgroup_nb_success / subgroup_nb_test
            subgroups_score.append(subgroup_score)

        group_score = np.mean(subgroups_score) * 100
        groups_score[group_name] = group_score

    return groups_score


def show_abx_matrix(ax, abx_matrix, phones, notation=None):
    if notation is None:
        notation = {}

    nb_phones = len(phones)
    ticks = np.arange(nb_phones)

    matrix_filter = abx_matrix[0] > 0
    matrix_norm = abx_matrix[0].copy()
    matrix_norm[matrix_filter] = (
        matrix_norm[matrix_filter] / abx_matrix[1][matrix_filter] * 100
    )

    im = ax.imshow(matrix_norm, vmin=0, vmax=100, cmap="cubehelix")
    ax.figure.colorbar(im, ax=ax)

    ticklabels = [notation[phone] if phone in notation else phone for phone in phones]
    ax.set_xticks(ticks, ticklabels)
    ax.set_yticks(ticks, ticklabels)
    ax.xaxis.tick_top()


def get_distance_signature(distance):
    signature_parts = []
    for features_type, feature_distance in distance.items():
        signature_parts.append(
            f"{features_type} x{feature_distance['weight']} ({feature_distance['metric']})"
        )
    return " + ".join(signature_parts)


def get_occlusions_metrics(phones, phones_indexes, datasets_ema, palate):
    phones_metrics = {}

    for phone in phones:
        phone_metrics = {
            "min_lips_distance": [],
            "min_lips_ema": [],
            "min_lips_index": [],
            "min_tongue_tip_distance": [],
            "min_tongue_tip_ema": [],
            "min_tongue_tip_index": [],
            "min_tongue_mid_distance": [],
            "min_tongue_mid_ema": [],
            "min_tongue_mid_index": [],
        }
        phones_metrics[phone] = phone_metrics

        for phone_index in phones_indexes[phone]:
            consonant_ema = datasets_ema[phone_index[0]][phone_index[1]][phone_index[2]]
            nb_frames = len(consonant_ema)

            lower_lip_ema = consonant_ema[:, 8:10]
            upper_lip_ema = consonant_ema[:, 10:12]
            lips_distance = np.sqrt(
                np.sum((lower_lip_ema - upper_lip_ema) ** 2, axis=1)
            )
            min_lips_distance = lips_distance.min()
            min_lips_frame = lips_distance.argmin()
            min_lips_ema = consonant_ema[min_lips_frame]
            min_lips_index = (
                phone_index[0],
                phone_index[1],
                phone_index[2].start + min_lips_frame,
            )

            phone_metrics["min_lips_distance"].append(min_lips_distance)
            phone_metrics["min_lips_ema"].append(min_lips_ema)
            phone_metrics["min_lips_index"].append(min_lips_index)

            tongue_tip_ema = consonant_ema[:, 2:4]
            tongue_mid_ema = consonant_ema[:, 4:6]
            repeated_palate = np.tile(palate, (nb_frames, 1, 1))

            tongue_tip_distances = np.sqrt(
                ((tongue_tip_ema[:, None, :] - repeated_palate) ** 2).sum(axis=-1)
            )
            min_tongue_tip_distance = tongue_tip_distances.min()
            min_tongue_tip_frame = tongue_tip_distances.min(axis=1).argmin()
            min_tongue_tip_ema = consonant_ema[min_tongue_tip_frame]
            min_tongue_tip_index = (
                phone_index[0],
                phone_index[1],
                phone_index[2].start + min_tongue_tip_frame,
            )

            phone_metrics["min_tongue_tip_distance"].append(min_tongue_tip_distance)
            phone_metrics["min_tongue_tip_ema"].append(min_tongue_tip_ema)
            phone_metrics["min_tongue_tip_index"].append(min_tongue_tip_index)

            tongue_mid_distances = np.sqrt(
                ((tongue_mid_ema[:, None, :] - repeated_palate) ** 2).sum(axis=-1)
            )
            min_tongue_mid_distance = tongue_mid_distances.min()
            min_tongue_mid_frame = tongue_mid_distances.min(axis=1).argmin()
            min_tongue_mid_ema = consonant_ema[min_tongue_mid_frame]
            min_tongue_mid_index = (
                phone_index[0],
                phone_index[1],
                phone_index[2].start + min_tongue_mid_frame,
            )

            phone_metrics["min_tongue_mid_distance"].append(min_tongue_mid_distance)
            phone_metrics["min_tongue_mid_ema"].append(min_tongue_mid_ema)
            phone_metrics["min_tongue_mid_index"].append(min_tongue_mid_index)

    return phones_metrics


def coil_distances_from_palate(coil_ema, palate):
    nb_frames, nb_dims = coil_ema.shape
    assert nb_dims == 2
    repeated_palate = np.tile(palate, (nb_frames, 1, 1))
    coil_distances = np.sqrt(
        ((coil_ema[:, None, :] - repeated_palate) ** 2).sum(axis=-1)
    ).min(axis=1)
    return coil_distances


def get_occlusions_indexes(
    phones, phones_indexes, detection_methods, datasets_ema, palate, occlusion_ceil=1
):
    occlusions_indexes = {}

    for phone in phones:
        phone_indexes = occlusions_indexes[phone] = []

        for phone_index in phones_indexes[phone]:
            phone_ema = datasets_ema[phone_index[0]][phone_index[1]][phone_index[2]]
            nb_frames = len(phone_ema)

            if (
                detection_methods[phone] == "tongue_tip"
                or detection_methods[phone] == "tongue_mid"
            ):
                consonant_coil = (
                    slice(2, 4)
                    if detection_methods[phone] == "tongue_tip"
                    else slice(4, 6)
                )
                coil_ema = phone_ema[:, consonant_coil]
                repeated_palate = np.tile(palate, (nb_frames, 1, 1))
                coil_distances = np.sqrt(
                    ((coil_ema[:, None, :] - repeated_palate) ** 2).sum(axis=-1)
                ).min(axis=1)
                occluded_frames = coil_distances < occlusion_ceil
            elif detection_methods[phone] == "lips":
                lips_distance = np.sqrt(
                    np.sum((phone_ema[:, 10:12] - phone_ema[:, 8:10]) ** 2, axis=1)
                )
                occluded_frames = np.zeros(nb_frames, dtype="bool")
                occluded_frames[1:-1] = (lips_distance[1:-1] < lips_distance[:-2]) & (
                    lips_distance[1:-1] < lips_distance[2:]
                )

            if occluded_frames.sum() < 2:
                continue
            occlusion_start = occluded_frames.argmax() + phone_index[2].start
            occlusion_stop = (
                nb_frames - occluded_frames[::-1].argmax() - 1 + phone_index[2].start
            )

            phone_indexes.append(
                (phone_index[0], phone_index[1], occlusion_start, occlusion_stop)
            )

    return occlusions_indexes
