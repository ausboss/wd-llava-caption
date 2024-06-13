import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as rt
import huggingface_hub
# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
# Files to download from the repos

LABEL_FILENAME = "selected_tags.csv"

# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]



class WaifuDiffusionTagger:
    def __init__(self, hf_token):
        self.HF_TOKEN = hf_token
        self.MODEL_FILENAME = "model.onnx"
        self.model_target_size = None
        self.last_loaded_repo = None
        self.tag_names = None
        self.rating_indexes = None
        self.general_indexes = None
        self.character_indexes = None
        self.model = None

    def load_labels(self, dataframe) -> list[str]:
        name_series = dataframe["name"]
        name_series = name_series.map(
            lambda x: x.replace("_", " ") if x not in kaomojis else x
        )
        tag_names = name_series.tolist()

        rating_indexes = list(np.where(dataframe["category"] == 9)[0])
        general_indexes = list(np.where(dataframe["category"] == 0)[0])
        character_indexes = list(np.where(dataframe["category"] == 4)[0])
        return tag_names, rating_indexes, general_indexes, character_indexes



    def mcut_threshold(self, probs):
        """
        Maximum Cut Thresholding (MCut)
        Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
        for Multi-label Classification. In 11th International Symposium, IDA 2012
        (pp. 172-183).
        """
        sorted_probs = probs[probs.argsort()[::-1]]
        difs = sorted_probs[:-1] - sorted_probs[1:]
        t = difs.argmax()
        thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
        return thresh


    def download_model(self, model_repo):
        csv_path = huggingface_hub.hf_hub_download(
            model_repo,
            LABEL_FILENAME,
            use_auth_token=self.HF_TOKEN,
        )
        model_path = huggingface_hub.hf_hub_download(
            model_repo,
            self.MODEL_FILENAME,
            use_auth_token=self.HF_TOKEN,
        )
        return csv_path, model_path


    def load_model(self, model_repo):
        if model_repo == self.last_loaded_repo:
            return

        csv_path, model_path = self.download_model(model_repo)

        tags_df = pd.read_csv(csv_path)
        sep_tags = self.load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        model = rt.InferenceSession(model_path)
        _, height, width, _ = model.get_inputs()[0].shape
        self.model_target_size = height

        self.last_loaded_repo = model_repo
        self.model = model

    def process_tags(self, tags_df):
        self.tag_names = tags_df['name'].map(lambda x: x.replace('_', ' ') if x not in kaomojis else x).tolist()
        self.rating_indexes = list(np.where(tags_df['category'] == 9)[0])
        self.general_indexes = list(np.where(tags_df['category'] == 0)[0])
        self.character_indexes = list(np.where(tags_df['category'] == 4)[0])

    def prepare_image(self, image):
        target_size = self.model_target_size

        # Ensure the image is in "RGB" mode for compatibility
        if image.mode != 'RGB':
            image = image.convert("RGB")

        # Create a canvas in "RGB" mode, since we don't need alpha compositing here
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        # Directly paste the image onto the canvas. No need for alpha_composite.
        canvas.paste(image)

        # Pad image to square
        max_dim = max(canvas.size)
        pad_left = (max_dim - canvas.size[0]) // 2
        pad_top = (max_dim - canvas.size[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(canvas, (pad_left, pad_top))

        # Resize if necessary
        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

        # Convert to numpy array and adjust for model input
        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]  # Convert RGB to BGR if necessary for your model

        return np.expand_dims(image_array, axis=0)


    def predict(
        self,
        image,
        model_repo,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
    ):
        self.load_model(model_repo)

        image = self.prepare_image(image)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = self.mcut_threshold(general_probs)

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        # Everything else is characters: pick any where prediction confidence > threshold
        character_names = [labels[i] for i in self.character_indexes]

        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = self.mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = (
            ", ".join(sorted_general_strings).replace("(", "\(").replace(")", "\)")
        )

        return sorted_general_strings, rating, character_res, general_res
