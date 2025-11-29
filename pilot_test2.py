from vlm.distractor_wrapper import distractor
from vlm.distractor_prompt import DISTRACTOR_PROMPT
from vlm.qwen3_swift import Qwen3VL

if __name__ == "__main__":

    default_video = "https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4"
    default_image = "https://ultralytics.com/images/bus.jpg"
    vlm_model = Qwen3VL()
    distractor_prompt = DISTRACTOR_PROMPT
    distractor = distractor(vlm_model, distractor_prompt)
    video_question = "who is the primary person in the video?"
    video_answer = "baby"
    print(distractor.generate(question=video_question, answer=video_answer, video=default_video))
    image_question = "where does this happend?"
    image_answer = "spain"
    print(distractor.generate(question=image_question, answer=image_answer, image=default_image))
