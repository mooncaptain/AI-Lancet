# AI-Lancet
AI-Lancet, backdoor removal

Architecture
├── BadNets 
│   └── YoutubeFace  
│       ├── 1.RegularTrigger
│   │   │      ├── AILancet
│   │   │      ├── backdoor_model_file
│   │   │      ├── BadNets
│   │   │      └── Unlearning
│       ├── 2.TransparentTrigger
│   │   │      ├── AILancet
│   │   │      ├── backdoor_model_file
│   │   │      ├── BadNets
│   │   │      └── Unlearning
│       └── Datasets

└── TrojanAttack  
    └── VGG-Face
	
	
#Requirements

Python==3.6.9
torch==1.10.1+cu113
torchvision==0.11.2+cu113
numpy==1.17.0
opencv-python==4.1.1.26
scipy==1.3.1
Pillow==8.4.0
	
###TrojanAttack
    #opensourced backdoor model，so we do not need to train a backdoor model.
    1. run debug_all_layer_delta_trigger.py，
       This code will reverse the trigger with 100 clean samples，then it goes through each layer to locate the EI neurons and save the results in \mask_back and \text_file.

    2. run debug_test_flip.py
       This code will flip EI neurons to remove the backdoor.

###BadNets		

###YoutubeFace--Regular Trigger
    #run our results
	1. ./BadNets/    #For original backdoor model, to measure the model accuracy and the backdoor attack success rate.
	    python3 model_test.py  
	2. ./AILancet/   #For the flipped model, to measure the model accuracy and the backdoor attack success rate.
	    python3 Neuron_flip.py
	3. ./Unlearning/ #For unlearning method, to meausre the model accuracy and the backdoor attack success rate.
	    python3 unlearn_model.py 

# train your own backdoor model and evaluate	
	You need delete "#!" in 3 files (train_backdoor_model.py，Restore_trigger.py ，Locate_EIneurons.py )
  1. BadNets #train the backdoor model
	   (Four kinds of triggers(1,2,3,4)，target_label is the target backdoored class）
	    python3 train_backdoor_model.py --trigger 1 --target_label 0  
	2. BadNets #to measure the backdoored model's model accuracy and backdoor attack success rate.
	    python3 model_test.py
	3.  AI-Lancet #Reverse the trigger
	    python3 Restore_trigger.py --trigger 1 --target_label 0
	4.  AI-Lancet #locate EI neurons
	    python3 Locate_EIneurons.py --trigger 1
	5.  AI-Lancet #Flip EI neurons and measure the flipped model 
		python3 Neuron_flip.py	
		

###YoutubeFace--Transparent Trigger
		
          Same as above.

