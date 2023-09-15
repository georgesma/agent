import numpy as np
from numpy import linalg
from sklearn.decomposition import PCA


def build_art_model(ema_frames):
    nb_coils = ema_frames.shape[1] // 2
    assert nb_coils == 6 or nb_coils == 7
    has_velum = nb_coils == 7

    jaw_data = ema_frames[:, 0:2]
    tongue_data = ema_frames[:, 2:8]
    lips_data = ema_frames[:, 8:12]
    if has_velum:
        velum_data = ema_frames[:, 12:14]

    ema_mean = ema_frames.mean(axis=0)

    jh_jaw_w, jh_values = extract_jh_jaw_weight(jaw_data)

    jh_tongue_w, tongue_data_jh_free = extract_jh_tongue_weight(jh_values, tongue_data)
    tbtd_tongue_w, tongue_data_tbtd_free = extract_tbtd_tongue_weight(
        tongue_data_jh_free
    )
    tt_tongue_w = extract_tt_tongue_weight(tongue_data_tbtd_free)

    jh_lips_w, lips_data_jh_free = extract_jh_lips_weight(jh_values, lips_data)
    lplh_lips_w = extract_lplh_lips_weight(lips_data_jh_free)

    vl_velum_w = extract_vl_velum_weight(velum_data) if has_velum else None

    art_to_ema_w = build_art_to_ema_matrix(
        jh_jaw_w,
        jh_tongue_w,
        tbtd_tongue_w,
        tt_tongue_w,
        jh_lips_w,
        lplh_lips_w,
        vl_velum_w,
    )
    ema_to_art_w = build_ema_to_art_matrix(
        jh_jaw_w,
        jh_tongue_w,
        tbtd_tongue_w,
        tt_tongue_w,
        jh_lips_w,
        lplh_lips_w,
        vl_velum_w,
    )

    art_model_params = {
        "ema_mean": ema_mean,
        "art_to_ema_w": art_to_ema_w,
        "ema_to_art_w": ema_to_art_w,
    }

    return art_model_params


def extract_jh_jaw_weight(jaw_data):
    jh_pca = PCA(n_components=1)
    jh_values = jh_pca.fit_transform(jaw_data)
    jh_jaw_w = jh_pca.components_
    return jh_jaw_w, jh_values


def extract_jh_tongue_weight(jh_values, tongue_data):
    tongue_mean = tongue_data.mean(axis=0)
    tongue_centered = tongue_data - tongue_mean
    jh_tongue_w = linalg.lstsq(jh_values, tongue_centered, rcond=None)[0]
    tongue_data_jh_free = tongue_centered - jh_values @ jh_tongue_w
    return jh_tongue_w, tongue_data_jh_free


def extract_tbtd_tongue_weight(tongue_data_jh_free):
    tbtd_pca = PCA(n_components=2)
    tbtd_values = tbtd_pca.fit_transform(tongue_data_jh_free[:, 2:6])
    tbtd_tongue_w = linalg.lstsq(tbtd_values, tongue_data_jh_free, rcond=None)[0]
    tongue_data_tbtd_free = tongue_data_jh_free - tbtd_values @ tbtd_tongue_w
    return tbtd_tongue_w, tongue_data_tbtd_free


def extract_tt_tongue_weight(tongue_data_tbtd_free):
    tt_pca = PCA(n_components=1)
    tt_pca.fit(tongue_data_tbtd_free)
    tt_tongue_w = tt_pca.components_
    return tt_tongue_w


def extract_jh_lips_weight(jh_values, lips_data):
    lips_mean = lips_data.mean(axis=0)
    lips_centered = lips_data - lips_mean
    jh_lips_w = linalg.lstsq(jh_values, lips_centered, rcond=None)[0]
    lips_data_jh_free = lips_centered - jh_values @ jh_lips_w
    return jh_lips_w, lips_data_jh_free


def extract_lplh_lips_weight(lips_data_jh_free):
    lx_pca = PCA(n_components=1)
    lx_values = lx_pca.fit_transform(lips_data_jh_free[:, (0, 2)])
    ly_pca = PCA(n_components=1)
    ly_values = ly_pca.fit_transform(lips_data_jh_free[:, (1, 3)])
    lxy_values = np.concatenate((lx_values, ly_values), axis=1)
    lplh_lips_w = linalg.lstsq(lxy_values, lips_data_jh_free, rcond=None)[0]
    return lplh_lips_w


def extract_vl_velum_weight(velum_data):
    vl_pca = PCA(n_components=1)
    vl_pca.fit(velum_data)
    vl_velum_w = vl_pca.components_
    return vl_velum_w


def build_art_to_ema_matrix(
    jh_jaw_w,
    jh_tongue_w,
    tbtd_tongue_w,
    tt_tongue_w,
    jh_lips_w,
    lplh_lips_w,
    vl_velum_w,
):
    has_velum = vl_velum_w is not None
    dim_art = 7 if has_velum else 6
    dim_ema = 14 if has_velum else 12

    art_to_ema_w = np.zeros((dim_art, dim_ema))

    art_to_ema_w[0, 0:2] = jh_jaw_w

    art_to_ema_w[0, 2:8] = jh_tongue_w
    art_to_ema_w[1:3, 2:8] = tbtd_tongue_w
    art_to_ema_w[3, 2:8] = tt_tongue_w

    art_to_ema_w[0, 8:12] = jh_lips_w
    art_to_ema_w[4:6, 8:12] = lplh_lips_w

    if has_velum:
        art_to_ema_w[6, 12:14] = vl_velum_w

    return art_to_ema_w


def build_ema_to_art_matrix(
    jh_jaw_w,
    jh_tongue_w,
    tbtd_tongue_w,
    tt_tongue_w,
    jh_lips_w,
    lplh_lips_w,
    vl_velum_w,
):
    has_velum = vl_velum_w is not None
    dim_art = 7 if has_velum else 6
    dim_ema = 14 if has_velum else 12

    # We build a first matrix allowing to find `jh` from jaw coordinates
    jh_jaw_w_inv = np.zeros((dim_ema, 1))
    jh_jaw_w_inv[0:2, :] = linalg.pinv(jh_jaw_w)

    # We build a second matrix allowing to remove the influence of `jh` on tongue and lips coordinates
    jh_tongue_lips_w = np.zeros((1, dim_ema))
    jh_tongue_lips_w[0, 2:8] = jh_tongue_w
    jh_tongue_lips_w[0, 8:12] = jh_lips_w

    # We build intermediate matrices allowing to find `tb`, `td`, `tt`, `lp`,
    # `lh` and `vl` from ema coordinates freed from `jh` influence
    tbtdtt_tongue_w = np.concatenate((tbtd_tongue_w, tt_tongue_w))
    tbtdtt_tongue_w_inv = linalg.pinv(tbtdtt_tongue_w)
    lplh_lips_w_inv = linalg.pinv(lplh_lips_w)
    if has_velum:
        vl_velum_w_inv = linalg.pinv(vl_velum_w)

    # We build a third matrix by assembling the intermediate matrices
    jh_free_ema_to_art_w = np.zeros((dim_ema, dim_art))
    jh_free_ema_to_art_w[:, 0:1] = jh_jaw_w_inv
    jh_free_ema_to_art_w[2:8, 1:4] = tbtdtt_tongue_w_inv
    jh_free_ema_to_art_w[8:12, 4:6] = lplh_lips_w_inv
    if has_velum:
        jh_free_ema_to_art_w[12:14, 6:7] = vl_velum_w_inv

    # We combine the main 3 matrices in a all-in-one "EMA coordinates to articulatory parameters" matrix
    ema_to_art_w = (
        np.eye(dim_ema) - jh_jaw_w_inv @ jh_tongue_lips_w
    ) @ jh_free_ema_to_art_w

    return ema_to_art_w


def art_to_ema(art_model_params, art_frames):
    return art_frames @ art_model_params["art_to_ema_w"] + art_model_params["ema_mean"]


def ema_to_art(art_model_params, ema_frames):
    return (ema_frames - art_model_params["ema_mean"]) @ art_model_params[
        "ema_to_art_w"
    ]
