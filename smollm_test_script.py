
from matching_functions import match_tensors_permute
from torch.utils.data import TensorDataset, DataLoader
from model_merger import ModelMerge
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy
from graphs.llama_graph import TransformerEncoderGraph

llama_modules = {"emb": "embed_tokens",
                "emb_ln": "input_layernorm",
                "q": "self_attn.q_proj",
                "k": "self_attn.k_proj",
                "v": "self_attn.v_proj",
                'attn_o': 'self_attn.o_proj',
                'post_attn_norm': 'post_attention_layernorm',
                'silu': 'mlp.act_fn',
                'up_proj': 'mlp.up_proj',
                'gate_proj': 'mlp.gate_proj',
                'down_proj': 'mlp.down_proj',
                'final_norm': 'norm'}


def make_graphified_models(model_name_list):
    models = []
    for model_name in model_name_list:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        # todo: add rotary embeddings
        # defaults merge the minimum so we need to add flags or change defaults
        # example = qk=True, merge_type='all', classifier=??
        graph = TransformerEncoderGraph(deepcopy(model), modules=llama_modules, num_layers=12)
        graph = graph.graphify()
        models.append(graph)
    return models

if __name__ == '__main__':

    model_name_list = ["HuggingFaceTB/SmolLM-135M-Instruct", "HuggingFaceTB/SmolLM-135M"]
    graph1, graph2 = make_graphified_models(model_name_list)

    if False:
        graph1.draw(save_path="/tmp/graph1.png", figsize=(60, 80))

    merge = ModelMerge(graph1, graph2, device="cpu")

    num_test_ex = 1
    input_ids = torch.randint(0, 100, (num_test_ex, 10))
    lens = torch.tensor([10]*num_test_ex).unsqueeze(1)
    dataloader = DataLoader(TensorDataset(input_ids, lens), batch_size=1)

    model3 = AutoModelForCausalLM.from_pretrained(model_name_list[0])
    model3.eval()

    # TODO: seq length currently is 8 instead of 10 because it is stripping some padding stuff that needs to be fixed
    merge.transform(model3, dataloader, transform_fn=match_tensors_permute, special_toks=True, res_type='first')
    # metric calculation is broken. Need to read paper.
    # TODO: using res_type='first' but should explore all and sep. Not sure what sep does.
    # TODO: We need to change apply_transformation_custom() to handle up and gate proj in MLP
    # TODO: We didn't modify unmerger but it didn't throw any errors so leaving it for now.
    # TODO: Check all the cases in the apply_transformation_custom() function.
    # TODO: Works without language model head. Need to fix that.
    # TODO: Extend merge to be for all layers.

    tokenizer = AutoTokenizer.from_pretrained(model_name_list[0])

    # Define prompt
    prompt = "Once upon a time in a distant galaxy,"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model3.device)

    # Generate output
    output = model3.generate(**inputs, max_length=100, temperature=0.7, top_p=0.9)

    # Decode and print output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)