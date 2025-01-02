'''
module : trailcam.py
Language : Python 3.x
email : andrew@openmarmot.com
github : https://github.com/openmarmot/indoor_trail_cam
notes : raspberry pi version - note this does not work as it exceeds the 8gb of memory the pi has
perhaps a smaller model could be substituted.
'''
# ref for the LLM : https://huggingface.co/MILVLG/imp-v1-3b

# -- import external packages --
# this stuff is needed for the img recognition model "imp"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

import cv2 # webcam stuff

from picamera2 import Picamera2 # interfacing with the pi camera

# -- import built in packages --
from datetime import datetime
import time

#---------------------------------------------------------------------------
def analyze_image(question,image,model,tokenizer):
    '''analyze an image with a LLM'''
    
    # prompt format for the v1.3 model
    system_prompt="A chat between a curious user and an assistant. The assistant gives helpful, detailed answers to the user's questions."
    text = system_prompt+' USER: <image>\n'+question+' ASSISTANT:'

    # prompt format for the new v1.5 models
    #system_prompt="<|im_start|>system\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.<|im_end|>"
    #text = system_prompt+"\n<|im_start|>user\n<image>\n"+question+"<|im_end|>\n<|im_start|>assistant"

    #image = Image.open(image_path)

    input_ids = tokenizer(text, return_tensors='pt').input_ids
    image_tensor = model.image_preprocess(image)

    #Generate the answer
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        images=image_tensor,
        use_cache=True)[0]
    
    result=tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    return result

#---------------------------------------------------------------------------
def load_llm():
    '''load the LLM model and tokenizer'''
    model_name="MILVLG/imp-v1-3b"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model,tokenizer  

#---------------------------------------------------------------------------
def get_image():
    '''capture a image'''


    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888'}))
    picam2.start()

    frame = picam2.capture_array()

    # No need for color conversion here since it's already in RGB
    pil_image = Image.fromarray(frame)

    return pil_image

    
#---------------------------------------------------------------------------
def run():
    question='Are there any cats in this image?'
    max_images=100
    image_count=0
    # load the llm components
    model,tokenizer=load_llm()

    # main loop 
    running=True
    while running:

        # take a picture
        image=get_image()

        # if image capture was successful
        if image:
            analysis=analyze_image(question,image,model,tokenizer)
            print(analysis)
            if 'Yes' in analysis:
                print('SUCCESS')
                time_stamp=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

                image.save('./images/cat_sighting_'+time_stamp+'.jpg')
                image_count+=1
                print('current image count: '+str(image_count))
                if image_count>max_images:
                    running=False
        else:
            print('image capture error !')

        # sleep for 30 seconds
        time.sleep(30)

    # cleanup
    model=None
    tokenizer=None

run()