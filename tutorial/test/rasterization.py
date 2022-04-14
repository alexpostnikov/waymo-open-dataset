import cv2
import numpy as np

MAX_PIXEL_VALUE = 255
N_ROADS = 21
road_colors = [int(x) for x in np.linspace(1, MAX_PIXEL_VALUE, N_ROADS).astype("uint8")]


def rasterize_batch(parsed,
        validate,
        crop_size=512,
        raster_size=224,
        shift=2 ** 9,
        magic_const=3,
        n_channels=11,
):
    tracks_to_predict = parsed["state/tracks_to_predict"].cpu().numpy()
    past_x = parsed["state/past/x"].cpu().numpy().reshape(-1,128,10)
    past_y = parsed["state/past/y"].cpu().numpy().reshape(-1,128,10)
    current_x = parsed["state/current/x"].cpu().numpy()
    current_y = parsed["state/current/y"].cpu().numpy()
    current_yaw = parsed["state/current/bbox_yaw"].cpu().numpy()
    past_yaw = parsed["state/past/bbox_yaw"].cpu().numpy().reshape(-1,128,10)
    past_valid = parsed["state/past/valid"].cpu().numpy().reshape(-1,128,10)
    current_valid = parsed["state/current/valid"].cpu().numpy()
    agent_type = parsed["state/type"].cpu().numpy()
    roadlines_coords = parsed["roadgraph_samples/xyz"].cpu().numpy().reshape(-1,20000,3)
    roadlines_types = parsed["roadgraph_samples/type"].cpu().numpy()
    roadlines_valid = parsed["roadgraph_samples/valid"].cpu().numpy()
    roadlines_ids = parsed["roadgraph_samples/id"].cpu().numpy()
    widths = parsed["state/current/width"].cpu().numpy()
    lengths = parsed["state/current/length"].cpu().numpy()
    agents_ids = parsed["state/id"].cpu().numpy()
    tl_states = parsed["traffic_light_state/current/state"].cpu().numpy()
    tl_ids = np.ones([tl_states.shape[0]])
    tl_valids = parsed["traffic_light_state/current/valid"].cpu().numpy()
    future_x = parsed["state/future/x"].cpu().numpy().reshape(-1,128,80)
    future_y = parsed["state/future/y"].cpu().numpy().reshape(-1,128,80)
    future_valid = parsed["state/future/valid"].cpu().numpy().reshape(-1,128,80)
    #scenario_id = np.ones([tl_states.shape[0]])#.decode("utf-8")
    rasters = []
    for bn in range(future_y.shape[0]):
        raster = rasterize(
            tracks_to_predict[bn], past_x[bn], past_y[bn],
            current_x[bn], current_y[bn], current_yaw[bn],
            past_yaw[bn], past_valid[bn], current_valid[bn],
            agent_type[bn], roadlines_coords[bn], roadlines_types[bn],
            roadlines_valid[bn], roadlines_ids[bn], widths[bn],
            lengths[bn], agents_ids[bn], tl_states[bn],
            tl_ids[bn], tl_valids[bn], future_x[bn],
            future_y[bn], future_valid[bn], True,
            crop_size=512, raster_size=224, shift=2 ** 9,
            magic_const=3, n_channels=11,
        )
        rasters.append(raster)
    return rasters


def rasterize(tracks_to_predict, past_x, past_y, current_x, current_y, current_yaw, past_yaw, past_valid, current_valid,
              agent_type, roadlines_coords, roadlines_types, roadlines_valid, roadlines_ids, widths, lengths, agents_ids,
              tl_states, tl_ids, tl_valids, future_x, future_y, future_valid, validate, crop_size=512, raster_size=224,
              shift=2 ** 9, magic_const=3, n_channels=11):
    GRES = []
    displacement = np.array([[raster_size // 4, raster_size // 2]]) * shift
    tl_dict = {"green": set(), "yellow": set(), "red": set()}

    # Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4,
    # Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
    for tl_state, tl_id, tl_valid in zip(
            tl_states.flatten(), tl_ids.flatten(), tl_valids.flatten()
    ):
        if tl_valid == 0:
            continue
        if tl_state in [1, 4, 7]:
            tl_dict["red"].add(tl_id)
        if tl_state in [2, 5, 8]:
            tl_dict["yellow"].add(tl_id)
        if tl_state in [3, 6]:
            tl_dict["green"].add(tl_id)

    XY = np.concatenate(
        (
            np.expand_dims(np.concatenate((past_x, current_x[:,np.newaxis]), axis=1), axis=-1),
            np.expand_dims(np.concatenate((past_y, current_y[:,np.newaxis]), axis=1), axis=-1),
        ),
        axis=-1,
    )

    GT_XY = np.concatenate(
        (np.expand_dims(future_x, axis=-1), np.expand_dims(future_y, axis=-1)), axis=-1
    )

    YAWS = np.concatenate((past_yaw, current_yaw[:,np.newaxis]), axis=1)

    agents_valid = np.concatenate((past_valid, current_valid[:,np.newaxis]), axis=1)

    roadlines_valid = roadlines_valid.reshape(-1)
    roadlines_coords = (
            roadlines_coords[:, :2][roadlines_valid > 0]
            * shift
            * magic_const
            * raster_size
            / crop_size
    )
    roadlines_types = roadlines_types[roadlines_valid > 0]
    roadlines_ids = roadlines_ids.reshape(-1)[roadlines_valid > 0]
    rasters = []
    for _, ( xy, current_val, val, _, yaw, agent_id, gt_xy, future_val, predict, ) \
            in enumerate(zip(XY, current_valid, agents_valid, agent_type, current_yaw.flatten(),
                              agents_ids, GT_XY, future_valid, tracks_to_predict.flatten())):
        if (not validate and future_val.sum() == 0) or (validate and predict == 0):
            continue
        if current_val == 0:
            continue

        RES_ROADMAP = (
                np.ones((raster_size, raster_size, 3), dtype=np.uint8) * MAX_PIXEL_VALUE
        )
        RES_EGO = [
            np.zeros((raster_size, raster_size, 1), dtype=np.uint8)
            for _ in range(n_channels)
        ]
        RES_OTHER = [
            np.zeros((raster_size, raster_size, 1), dtype=np.uint8)
            for _ in range(n_channels)
        ]

        xy_val = xy[val > 0]
        if len(xy_val) == 0:
            continue

        unscaled_center_xy = xy_val[-1].reshape(1, -1)
        center_xy = unscaled_center_xy * shift * magic_const * raster_size / crop_size
        rot_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ]
        )

        centered_roadlines = (roadlines_coords - center_xy) @ rot_matrix + displacement
        centered_others = (
                                  XY.reshape(-1, 2) * shift * magic_const * raster_size / crop_size
                                  - center_xy
                          ) @ rot_matrix + displacement
        centered_others = centered_others.reshape(128, n_channels, 2)
        centered_gt = (gt_xy - unscaled_center_xy) @ rot_matrix

        unique_road_ids = np.unique(roadlines_ids)
        for road_id in unique_road_ids:
            if road_id >= 0:
                roadline = centered_roadlines[roadlines_ids == road_id]
                road_type = roadlines_types[roadlines_ids == road_id].flatten()[0]

                road_color = road_colors[road_type]
                for c, rgb in zip(
                        ["green", "yellow", "red"],
                        [
                            (0, MAX_PIXEL_VALUE, 0),
                            (MAX_PIXEL_VALUE, 211, 0),
                            (MAX_PIXEL_VALUE, 0, 0),
                        ],
                ):
                    if road_id in tl_dict[c]:
                        road_color = rgb

                RES_ROADMAP = cv2.polylines(
                    RES_ROADMAP,
                    [roadline.astype(int)],
                    False,
                    road_color,
                    shift=9,
                )

        unique_agent_ids = np.unique(agents_ids)

        is_ego = False
        self_type = 0
        _tmp = 0
        for other_agent_id in unique_agent_ids:
            other_agent_id = int(other_agent_id)
            if other_agent_id < 1:
                continue
            if other_agent_id == agent_id:
                is_ego = True
                self_type = agent_type[agents_ids == other_agent_id]
            else:
                is_ego = False

            _tmp += 1
            agent_lane = centered_others[agents_ids == other_agent_id][0]
            agent_valid = agents_valid[agents_ids == other_agent_id]
            agent_yaw = YAWS[agents_ids == other_agent_id]

            agent_l = lengths[agents_ids == other_agent_id]
            agent_w = widths[agents_ids == other_agent_id]

            for timestamp, (coord, valid_coordinate, past_yaw,) in enumerate(
                    zip(
                        agent_lane,
                        agent_valid.flatten(),
                        agent_yaw.flatten(),
                    )
            ):
                if valid_coordinate == 0:
                    continue
                box_points = (
                        np.array(
                            [
                                -agent_l,
                                -agent_w,
                                agent_l,
                                -agent_w,
                                agent_l,
                                agent_w,
                                -agent_l,
                                agent_w,
                            ]
                        )
                        .reshape(4, 2)
                        .astype(np.float32)
                        * shift
                        * magic_const
                        / 2
                        * raster_size
                        / crop_size
                )

                box_points = (
                        box_points
                        @ np.array(
                    (
                        (np.cos(yaw - past_yaw), -np.sin(yaw - past_yaw)),
                        (np.sin(yaw - past_yaw), np.cos(yaw - past_yaw)),
                    )
                ).reshape(2, 2)
                )

                _coord = np.array([coord])

                box_points = box_points + _coord
                box_points = box_points.reshape(1, -1, 2).astype(np.int32)

                if is_ego:
                    cv2.fillPoly(
                        RES_EGO[timestamp],
                        box_points,
                        color=MAX_PIXEL_VALUE,
                        shift=9,
                    )
                else:
                    cv2.fillPoly(
                        RES_OTHER[timestamp],
                        box_points,
                        color=MAX_PIXEL_VALUE,
                        shift=9,
                    )

        raster = np.concatenate([RES_ROADMAP] + RES_EGO + RES_OTHER, axis=2)
        rasters.append(raster)
    return rasters
