import PIL
from cog import BasePredictor, Input, Path
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

CACHE_DIR = "model-cache"
MODEL_ID = "THUDM/chatglm-6b"


model_urls = {
    "textcaps": "google/pix2struct-textcaps-large",  # Finetuned on TextCaps
    "screen2words": "google/pix2struct-screen2words-large",  # Finetuned on Screen2Words
    "widgetcaption": "google/pix2struct-widget-captioning-large",  # Finetuned on Widget Captioning (captioning a UI component on a screen)
    "infographics": "google/pix2struct-infographics-vqa-large",  # Infographics
    "docvqa": "google/pix2struct-docvqa-large",  # Visual question answering
    "ai2d": "google/pix2struct-ai2d-large",  # Scienfic diagram
}

for model, model_path in model_urls.items():
    processor = Pix2StructProcessor.from_pretrained(model_path, cache_dir=CACHE_DIR)
    model = Pix2StructForConditionalGeneration.from_pretrained(
        model_path, cache_dir=CACHE_DIR
    )


class Predictor(BasePredictor):
    def setup(self):
        self.models = {}
        for model_name in model_urls.keys():
            print(f"Loading {model_name} from {model_urls[model_name]}")
            model = Pix2StructForConditionalGeneration.from_pretrained(
                model_urls[model_name], cache_dir=CACHE_DIR, local_files_only=True
            )
            processor = Pix2StructProcessor.from_pretrained(
                model_urls[model_name], cache_dir=CACHE_DIR, local_files_only=True
            )
            self.models[model_name] = (model, processor)

    def predict(
        self,
        image: Path = Input(description="Input Image."),
        task: str = Input(
            description="Choose a task.",
            choices=model_urls.keys(),
            default="screen2words",
        ),
        text: str = Input(description="Input text."),
    ) -> str:
        model, processor = self.models[task]
        model.to("cuda")

        image = PIL.Image.open(image)
        if processor.image_processor.is_vqa:
            print(f"Adding prompt for VQA model: '{text}'")
            inputs = processor(image, return_tensors="pt", text=text).to("cuda")
        else:
            inputs = processor(image, return_tensors="pt").to("cuda")
        predictions = model.generate(**inputs)

        return processor.decode(predictions[0], skip_special_tokens=True)
