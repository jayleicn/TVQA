### TVQA Data

#### Annotations

TVQA contains 3 available splits, as listed below:
| File | #QAs | Usage |
|---|---|---|
| [tvqa_train.jsonl](tvqa_train.jsonl) | 122,039 | Model training |
| [tvqa_val.jsonl](tvqa_val.jsonl) | 15,253 | Hyperparameter tuning |
| [tvqa_test_public.jsonl](tvqa_test_public.jsonl) | 7,623 | Model testing. Labels are not released for this set, please upload your predictions to the server for testing. |

Note, the original test set described in the TVQA paper is split into two subsets, test-public (7,623 QAs) and test-challenge (7,630 QAs). The test-public set is used for paper publication, the test-challenge set is reserved for future use.

Each line of these files can be loaded as a JSON object, containing the following entries:
| Key | Type | Description |
|---|---|---|
| qid | int | question id |
| q | str | question |
| a0, ..., a4 | str | multiple choice answers |
| answer_idx | int | answer index, this entry does not exist for test_public |
| ts | str | timestamp annotation. e.g. '76.01-84.2' denotes the localized span starts at 76.01 seconds, ends at 84.2 seconds. |
| vid_name | str | name of the video clip accompanies the question. The videos are named following the format '{show_name_abbr}_s{season_number}e{episode_number}_seg{segment_number}_clip_{clip_number}' e.g. 'friends_s06e12_seg02_clip_16' denotes the video clip is from season 6 episode 12 of the TV show 'Friends', it is the 16th clip of the 2nd segment. An episode typically has two segments, divided by the opening song. Also, note video clips for 'The Big Bang Theory' do not have '{show_name_abbr}' in their 'vid_name'. |
| show_name | str | name of the TV show |

A sample of the QA is shown below:
```json
{
	"a0": "A martini glass",
	"a1": "Nachos",
	"a2": "Her purse",
	"a3": "Marshall's book",
	"a4": "A beer bottle",
	"answer_idx": 4,
	"q": "What is Robin holding in her hand when she is talking to Ted about Zoey?",
	"qid": 7,
	"ts": "1.21-8.49",
	"vid_name": "met_s06e22_seg01_clip_02"
}
```


#### Subtitles
The subtitles are preprocessed and are taken from [this link](https://github.com/jayleicn/TVRetrieval/blob/master/data/tvqa_preprocessed_subtitles.jsonl).


#### Video Frames 

Download link: tvqa_video_frames_fps3_hq.tar.gz [43GB], please fill out this form first. We will review and respond to your request in 7 days. If you did not receive any response from us, please contact jielei [@] cs.unc.edu

The video frames are extracted at 3 frames per second (FPS), we show a sample of them below. To obtain the frames, please fill out the form first. You will be required to provide information about you and your advisor, as well as sign our agreement. The download link for the video frames will be sent to you in around a week if your form is valid. Please do not share the video frames with others.

![video_frame](../imgs/video_frame_example.png)


