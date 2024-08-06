import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


class Florence2Model:
    def __init__(self, model_id="microsoft/Florence-2-base") -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def run_example(self, image, task_prompt, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.shape[1], image.shape[0]),
        )
        return parsed_answer

    def convert_to_od_format(self, data):
        bboxes = data.get("bboxes", [])
        labels = data.get("bboxes_labels", []) or data.get("labels", [])

        od_results = {"bboxes": bboxes, "labels": labels}

        return od_results

    def predict(self, image, task_prompt="<OD>", text_input=None):
        results = self.run_example(image, task_prompt, text_input)
        return self.convert_to_od_format(results[task_prompt])

    def draw_bbox_cv2(self, frame, bboxes, labels, colour=(255, 0, 0)
                      , thickness=2):
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                colour,
                thickness,
            )
