# OmniDy_Bench

### Contents

- [Installation](#installation)
- [Dataset](#dataset)

## Installation
```
pip install ai2thor prior openai
```

Run evaluation on random agent
```bash
python3 runner.py -a agents.random_agent -c ./configs/procthor.yaml -d ./dataset -p train -l static -i 1 -n 1
```

This command runs inference with the random agent over the debug split. You can pass the args (`--train`, `--val`, and/or `--test`) or `--submission` instead to run this agent on other splits.


After installing and running the demo, you should see log messages that resemble the following:
```
2025-09-29 17:04:58,089 [INFO] procthor_challenge.challenge - Task Start instruction:What texture is the Laptop in Bedroom ? scene:train_1 initial_position:{'x': 5.0, 'y': 0.9009993672370911, 'z': 15.25} rotation:0
INFO:procthor_challenge.challenge:Task Start instruction:What texture is the Laptop in Bedroom ? scene:train_1 initial_position:{'x': 5.0, 'y': 0.9009993672370911, 'z': 15.25} rotation:0
[{'x': 5.0, 'y': 0.00650787353515625, 'z': 15.25}, {'x': 3.5799999237060547, 'y': 0.00650787353515625, 'z': 15.75999927520752}, {'x': 2.75, 'y': 0.00650787353515625, 'z': 15.819999694824219}, {'x': 2.5999999046325684, 'y': 0.00650787353515625, 'z': 15.819999694824219}, {'x': 2.559999942779541, 'y': 0.00650787353515625, 'z': 15.799999237060547}, {'x': 2.4700000286102295, 'y': 0.00650787353515625, 'z': 15.75}, {'x': 2.3899998664855957, 'y': 0.00650787353515625, 'z': 15.569999694824219}, {'x': 2.3899998664855957, 'y': 0.00650787353515625, 'z': 15.5}, {'x': 2.450000047683716, 'y': 0.00650787353515625, 'z': 15.359999656677246}, {'x': 2.4800000190734863, 'y': 0.00650787353515625, 'z': 15.309999465942383}, {'x': 3.359999895095825, 'y': 0.00650787353515625, 'z': 13.889999389648438}, {'x': 3.490000009536743, 'y': 0.00650787353515625, 'z': 13.819999694824219}, {'x': 7.0, 'y': 0.00650787353515625, 'z': 12.069999694824219}, {'x': 11.219999313354492, 'y': 0.00650787353515625, 'z': 11.529999732971191}, {'x': 11.40999984741211, 'y': 0.00650787353515625, 'z': 11.529999732971191}, {'x': 11.539999961853027, 'y': 0.00650787353515625, 'z': 11.59999942779541}, {'x': 11.639999389648438, 'y': 0.00650787353515625, 'z': 11.809999465942383}, {'x': 11.75, 'y': 0.00650787353515625, 'z': 14.25}]
2025-09-29 17:04:58,423 [INFO] procthor_challenge.challenge - Agent action: LookUp
INFO:procthor_challenge.challenge:Agent action: LookUp
total_step: 0 action success: True total time: 0.09546089172363281
2025-09-29 17:04:58,641 [INFO] procthor_challenge.challenge - Agent action: LookDown
INFO:procthor_challenge.challenge:Agent action: LookDown
total_step: 1 action success: True total time: 0.07021379470825195
2025-09-29 17:04:58,835 [INFO] procthor_challenge.challenge - Agent action: LookUp
INFO:procthor_challenge.challenge:Agent action: LookUp
total_step: 2 action success: True total time: 0.06385231018066406

```

## Dataset
Download the dataset and put into the dataset/ dir.
```
git lfs install
git clone https://huggingface.co/datasets/yhncao/UniDy_Bench
```  

The dataset is divided into the following splits:
### Nav  

| Split |Level |Scene| Episodes | Files |
| ----- |:-----|:-----|:-----:|-----|
| Train |Static|500| 1000k | `dataset/static/episode/train/{scene_id}/nav.json.gz`|
| Val   |Static|200| 400k | `dataset/static/episode/val/{scene_id}/nav.json.gz` |  
| Test  |Static|100| 200k | `dataset/static/episode/test/{scene_id}/nav.json.gz`|

### A-EQA 

| Split |Level |Scene| Episodes | Files |
| ----- |:-----|:-----|:-----:|-----|
| Train |Static|500| 153417 | `dataset/static/episode/train/{scene_id}/eqa.json.gz`|
| Val   |Static|200| 61088 | `dataset/static/episode/val/{scene_id}/eqa.json.gz` |  
| Test  |Static|100| 23948 | `dataset/static/episode/test/{scene_id}/eqa.json.gz`|

where each file is a compressed json file corresponding to a list of
dictionaries. Each element of the list corresponds to a single episode of object navigation.

<details>
<summary><b>Episode Structure</b></summary>
<p>
Here is an example of the structure of a single episode in our training set.

```javascript
Nav Episode:
{
    'instruction': [
            'Find the CD which is next to the CreditCard', 
            'Find the Watch', 
            'Find the Bowl showing in the image.', 
            'Find the silver, metal Spoon', 
            'Find the Bowl showing in the image.', 
            'Find the Dresser showing in the image.', 
            'Find the TeddyBear showing in the image.', 
            'I need something to keep track of the time.', 
            'Find the RemoteControl', 
            'I need paper towels for cleaning.'
    ], 
    'image_name': [
        None, 
        None, 
        'Bowl|surface|4|37_1.png', 
        None, 
        'Bowl|surface|5|58_1.png', 
        'Dresser|8|1_1.png', 
        'TeddyBear|surface|10|109_2.png',
        None, 
        None, 
        None
    ], 
    'level': 'static', 
    'id': [
        'CD|surface|10|91', 
        None, 
        'Bowl|surface|4|37', 
        'Spoon|surface|6|7', 
        'Bowl|surface|5|58', 
        'Dresser|8|1', 
        'TeddyBear|surface|10|109', 
        None, 
        None, 
        None
    ], 
    'task_type': [2, 1, 5, 4, 5, 5, 5, 3, 1, 3], 
    'initial_horizon': 0, 
    'initial_orientation': 0, 
    'initial_position': {'x': 13.0, 'y': 0.9009993672370911, 'z': 1.75}, 
    'scene_type': 'test', 
    'scene_id': 41, 
    'object_type': [
        None, 
        'Watch', 
        None, 
        None, 
        None, 
        None, 
        None,  
        ['Watch', 'AlarmClock', 'CellPhone'], 
        'RemoteControl', 
        ['PaperTowelRoll']
    ]
}
```  

```javascript  
A-EQA Episode:
{
    'time': 75, 
    'instruction': ['How many ToiletPapers are there in the Bathroom?'], 'level': 'static', 
    'task_type': [10], 
    'id': ['ToiletPaper|surface|9|88'], 
    'initial_horizon': 0, 
    'initial_orientation': 0, 
    'initial_position': {'x': 9.5, 'y': 0.9009993672370911, 'z': 1.25}, 'scene_type': 'test', 
    'scene_id': 41, 
    'object_type': [None], 
    'answer': [{'0': 1}]
}
```


All the episodes for each split (train/val/test) can be found within `dataset/`.  Configuration parameters for the environment can be found within `configs/procthor.yaml`.

