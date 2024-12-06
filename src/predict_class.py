import torch
from transformers import AutoTokenizer

from src.config import FilePath
from src.resources import load_model_from_disc


def predict_class(
    text: str,
    model_path: str = FilePath.model_2_classes,
    max_length: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> int:
    """
    Predict the class of a given text using a trained BERT model.

    Args:
        text (str): The input text to classify.
        model_path (str): Path to the trained model.
        tokenizer_path (str): Path to the tokenizer corresponding to the trained model.
        max_length (int): Maximum sequence length for tokenization. Defaults to 128.
        device (str): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda' if available.

    Returns:
        int: The predicted class label.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    model = load_model_from_disc(model_path)
    model.to(device)
    model.eval()

    # Tokenize the input text
    encoded_input = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    # Move input data to the device
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)

    # Return the predicted class
    return int(predictions.item())
