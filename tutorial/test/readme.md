## How to Download dataset:

```bash
pip install gsutil
pip install gcloud

gcloud auth login
```
goto [waymo files](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_1_0;tab=objects?prefix=&forceOnObjectsSortingFiltering=false) page
find folder\file to download, e.g. :
```
gsutil -m cp -r \
  "gs://waymo_open_dataset_motion_v_1_1_0/uncompressed" \
  .
```
## Submission Creation Pseudo-code:
```python
from waymo_open_dataset.protos import motion_submission_pb2

motion_challenge_submission = motion_submission_pb2.MotionChallengeSubmission()
motion_challenge_submission.account_name = "alex.postnikov@skolkovotech.ru"
motion_challenge_submission.submission_type = (
        motion_submission_pb2.MotionChallengeSubmission.SubmissionType.MOTION_PREDICTION
    )
motion_challenge_submission.unique_method_name = "iab"
#predictions = {scenario_id: [["aid": agent_id, "conf": conf, "pred": traj], [{"aid": aid, "conf": conf, "pred": p}]]}

selector = np.arange(4, 80 + 1, 5)
for scenario_id, data in tqdm(predictions):
    scenario_predictions = motion_challenge_submission.scenario_predictions.add()
    scenario_predictions.scenario_id = scenario_id
    prediction_set = scenario_predictions.single_predictions

    for d in data:
        predictions = prediction_set.predictions.add()
        predictions.object_id = int(d["aid"])

        for i in np.argsort(-d["conf"]):
            scored_trajectory = predictions.trajectories.add()
            scored_trajectory.confidence = d["conf"][i]
            trajectory = scored_trajectory.trajectory
            p = d["pred"][selector, i]  #@ rot_matrix + d["center"]
            trajectory.center_x.extend(p[:, 0])
            trajectory.center_y.extend(p[:, 1])

with open("file.pb", "wb") as f:
    f.write(motion_challenge_submission.SerializeToString())
```
    
    