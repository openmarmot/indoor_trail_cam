'''
module : trailcam.py
Language : Python 3.x
email : andrew@openmarmot.com
github : https://github.com/openmarmot/indoor_trail_cam
notes : take images with a laptop webcam and use a LLM to determine which ones to save
'''
# ref for the LLM : https://huggingface.co/MILVLG/imp-v1-3b

# -- import external packages --
# this stuff is needed for the img recognition model "imp"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

import cv2 # webcam stuff

# -- import built in packages --
from datetime import datetime
import time

#---------------------------------------------------------------------------
def analyze_image(question,image,model,tokenizer):
    '''analyze an image with a LLM'''
    #Set inputs
    system_prompt="A chat between a curious user and an assistant. The assistant gives helpful, detailed answers to the user's questions."
    text = system_prompt+' USER: <image>\n'+question+' ASSISTANT:'
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
    model = AutoModelForCausalLM.from_pretrained(
        "MILVLG/imp-v1-3b", 
        torch_dtype=torch.float32, #note this is optimized for cpu not gpu
        device_map="auto",
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True)
    return model,tokenizer  



#---------------------------------------------------------------------------
def get_image():
    '''capture a image'''
    # Capture from the default camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Convert the image from BGR to RGB (OpenCV uses BGR by default)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the NumPy array to a Pillow Image that IMP can understand
        pil_image = Image.fromarray(rgb_image)
        return pil_image
    else:
        return None
    
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

                image.save('./images/cat_sighting'+time_stamp+'.jpg')
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