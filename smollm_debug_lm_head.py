from matching_functions import match_tensors_permute
from torch.utils.data import TensorDataset, DataLoader
from model_merger import ModelMerge
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy
from graphs.llama_graph import TransformerEncoderGraph
from smollm_test_script import llama_modules
import argparse
import numpy as np
# Test perplexity of each model
test_text = """Here's a function to compute the perplexity or loss of the generated output using a language model. 
This function assumes that you have a trained language model that can return log probabilities for a given sequence. 
I'll write the function in Python using PyTorch, assuming you're using a transformer-based model like GPT."""


def make_graphified_models(model_name_list, num_layers=30, add_head=True):
    models = []
    for model_name in model_name_list:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        # todo: add rotary embeddings
        # defaults merge the minimum so we need to add flags or change defaults
        # example = qk=True, merge_type='all', classifier=??
        graph = TransformerEncoderGraph(deepcopy(model), modules=llama_modules, num_layers=num_layers, add_lm_head=add_head)
        graph = graph.graphify()
        models.append(graph)
    return models

if __name__ == '__main__':

    # add commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_type", type=str, choices=["random", "text"], default="text")
    parser.add_argument("--no-add_head", dest="add_head", action="store_false", default=True,
                   help="Enabled by default, disable with --no-add_head") 
    parser.add_argument("--test_parameter_change", action="store_true", default=False,
                   help="Test if parameters changed")
    parser.add_argument("--test_base_loss", action="store_true", default=False,
                   help="Test base loss")
    args = parser.parse_args()
    

    model_name_list = ["HuggingFaceTB/SmolLM-135M-Instruct", "HuggingFaceTB/SmolLM-135M"]
    tokenizer = AutoTokenizer.from_pretrained(model_name_list[0])
    num_test_ex = 10
    if args.input_type == "text":
        with open("random_text.txt", "r") as f:
            text = f.read()
        text = text.split("\n")
        tokenized_text = tokenizer(text)
        input_ids = torch.tensor([line[:15] for line in tokenized_text['input_ids']][:num_test_ex])
        lens = torch.tensor([input_ids.shape[1]]*num_test_ex).unsqueeze(1)
        dataloader = DataLoader(TensorDataset(input_ids, lens), batch_size=1)
    else: # random
        input_ids = torch.randint(0, tokenizer.vocab_size, (num_test_ex, 15))
        lens = torch.tensor([input_ids.shape[1]]*num_test_ex).unsqueeze(1)
        dataloader = DataLoader(TensorDataset(input_ids, lens), batch_size=1)

    for num_layers in [1, 4, 8, 15, 25, 30]:
        graph1, graph2 = make_graphified_models(model_name_list, num_layers, args.add_head)
        merge = ModelMerge(graph1, graph2, device="cpu")

        model3 = AutoModelForCausalLM.from_pretrained(model_name_list[0])
        model3.eval()

        if args.test_base_loss:
            inputs = tokenizer(test_text, return_tensors='pt')
            input_ids = inputs['input_ids'].to(model3.device)
            outputs = model3(input_ids, labels=input_ids)
            loss = outputs.loss
            print(f"Base Model Loss: {loss}")
            exit()

        if args.test_parameter_change:
            weight_norms = {}
            for name, param in model3.named_parameters():
                weight_norms[name] = param.data.norm().item()

        merge.transform(model3, dataloader, transform_fn=match_tensors_permute, special_toks=True, res_type='first', permute_heads=False)
        # TODO: test permute heads = True

        if args.test_parameter_change:
            print("Checking if parameters changed")
            # Check which weights were changed
            for name, param in model3.named_parameters():
                #print((np.abs(weight_norms[name] - param.data.norm().item())) / weight_norms[name], name, weight_norms[name], param.data.norm().item())
                if weight_norms[name] == param.data.norm().item():
                    print(f"##### Uh oh! {name} not changed")

        with torch.no_grad():
            inputs = tokenizer(test_text, return_tensors='pt')
            input_ids = inputs['input_ids'].to(model3.device)
            outputs = model3(input_ids, labels=input_ids)
            loss = outputs.loss
            # perplexity = torch.exp(loss)
            print(f"Loss with {num_layers} layers merged: {loss}") # || {perplexity}")
            # sample decode
            prompt = "Once upon a time in a distant galaxy,"
            inputs = tokenizer(prompt, return_tensors="pt").to(model3.device)
            output = model3.generate(**inputs, max_new_tokens=40)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(generated_text)

