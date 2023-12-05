from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel
import argparse


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="the language model of interest on HuggingFace")
    argParser.add_argument("-d", "--device", default = -1, help="device ID, -1 for CPU, >=0 for GPU ID")

    args = argParser.parse_args()
    model = args.model
    device = int(args.device)

    ##################### This Part is Added #####################

    import torch
    from transformers import GPT2Tokenizer, GPT2Model
    import numpy as np

    # Custom model class for binary classification
    class GPT2ForBinaryClassification(GPT2Model):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = 2
            self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

        def forward(self, input_ids, attention_mask=None, labels=None):
            outputs = super().forward(input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state

            # Apply mean pooling on the sequence of hidden states
            mean_last_hidden_state = torch.mean(last_hidden_states, dim=1)
            logits = self.classifier(mean_last_hidden_state)

            return logits

    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    

    # Load the fine-tuned binary classification model
    binary_model = GPT2ForBinaryClassification.from_pretrained('./gpt2_finetuned')

    # Load the GPT-2 model with a language modeling head
    lm_model = GPT2LMHeadModel.from_pretrained('gpt2', use_cache=False)

    # Transfer the base GPT-2 model weights from the binary model to the LM model
    lm_model_state_dict = lm_model.state_dict()
    binary_model_state_dict = {k: v for k, v in binary_model.state_dict().items() if k in lm_model_state_dict}
    lm_model_state_dict.update(binary_model_state_dict)
    lm_model.load_state_dict(lm_model_state_dict)

    # Set the model to evaluation mode
    lm_model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    ##################### The End #####################


    generator = pipeline("text-generation", model = lm_model, tokenizer = tokenizer, device = device, max_new_tokens = 100)
    result = generator("Do you want to build a snowman?")
    print(result[0]["generated_text"])
    print("success!")