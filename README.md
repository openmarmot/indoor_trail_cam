# indoor_trail_cam
using  the IMP LLM to take pictures of my cat

## about
I wanted to explore using the excellent IMP AI Image recognition model with a laptop web cam.
This turned out to be fairly easy, and the IMP model is surprising good at this task - even for 
relatively poor images from my laptop webcam.

This repo is more of a proof of concept than anything serious, but this could extended to be a 
pretty amazing trailcam or security camera system.

![screenshot](/screenshots/ai_output.png "IMG output")

## usage 
-(linux) sh run.sh

## troubleshooting  
- If you get module import errors and already have a venv, try deleting the venv and re-running run.sh  

### References
- [IMP LLM on HuggingFace](https://huggingface.co/MILVLG/imp-v1-3b)