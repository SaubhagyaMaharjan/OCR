import streamlit as st
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re
import json


# Load Donut model and processor
@st.cache_resource
def load_model():
    processor = DonutProcessor.from_pretrained("katanaml-org/invoices-donut-model-v1")
    model = VisionEncoderDecoderModel.from_pretrained(
        "katanaml-org/invoices-donut-model-v1"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device


processor, model, device = load_model()


# Clean and parse functions
def clean_donut_output(text):
    """
    Cleans up common Donut model errors such as:
    - Junk before first tag
    - Removing repeated commas
    - Removing whitespaces
    """

    text = re.sub(r"^[^<]*", "", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_nested_tags(text):
    top_level_tags = re.findall(r"<(s_[^>]+)>(.*?)</\1>", text, re.DOTALL)
    result = {}
    for section_tag, section_content in top_level_tags:
        inner_tags = re.findall(r"<(s_[^>]+)>(.*?)</\1>", section_content, re.DOTALL)
        for key, value in inner_tags:
            value = value.strip()
            if key in result:
                if isinstance(result[key], list):
                    result[key].append(value)
                else:
                    result[key] = [result[key], value]
            else:
                result[key] = value
    return result


def build_structured_json(parsed):
    """
    Converts nested <s_*> fields inside header, items, and summary into
    a clean JSON format matching the ground truth schema.
    """

    return {
        "header": {
            "invoice_no": parsed.get("s_invoice_no"),
            "invoice_date": parsed.get("s_invoice_date"),
            "seller": parsed.get("s_seller"),
            "client": parsed.get("s_client"),
            "seller_tax_id": parsed.get("s_seller_tax_id"),
            "client_tax_id": parsed.get("s_client_tax_id"),
            "iban": parsed.get("s_iban"),
        },
        "items": [
            {
                "item_desc": parsed.get("s_item_desc"),
                "item_qty": parsed.get("s_item_qty"),
                "item_net_price": parsed.get("s_item_net_price"),
                "item_net_worth": parsed.get("s_item_net_worth"),
                "item_vat": parsed.get("s_item_vat"),
                "item_gross_worth": parsed.get("s_item_gross_worth"),
            }
        ],
        "summary": {
            "total_net_worth": parsed.get("s_total_net_worth"),
            "total_vat": parsed.get("s_total_vat"),
            "total_gross_worth": parsed.get("s_total_gross_worth"),
        },
    }


# Main Streamlit interface
st.set_page_config(
    page_title="An Automated Information Extraction System", layout="wide"
)
st.title("An Automated Information Extraction System")
st.markdown(
    """
    An automated information extraction system that uses **AI** and **OCR** technologies to parse and extract 
    key-value pairs and tabular data from invoices of varying formats and structures.
    """
)
# File uploader
# Upload and show image
image_file = st.file_uploader("Upload Invoice Image", type=["jpg", "jpeg", "png"])

if image_file:
    image = Image.open(image_file).convert("RGB")

    # Center image using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Invoice", width=500)

    if st.button("Extract JSON"):
        with st.spinner("Extracting JSON.... please wait"):
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
            decoder_input_ids = processor.tokenizer(
                "<s_cord-v2>", add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(device)

            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=1024,
                    early_stopping=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            raw_output = processor.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )[0].strip()
            cleaned_output = clean_donut_output(raw_output)
            parsed = extract_nested_tags(raw_output)
            structured_json = build_structured_json(parsed)

        st.success("Extraction complete!")
        st.subheader("Extracted JSON")
        st.json(structured_json)
