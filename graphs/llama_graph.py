from graphs.base_graph import BIGGraph, NodeType
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader

from model_merger import ModelMerge
from matching_functions import match_tensors_permute
from copy import deepcopy

class TransformerEncoderGraph(BIGGraph):
    
    def __init__(self, model,
                 modules,
                 layer_name='layers', # for transformer
                 enc_prefix='model',
                 merge_type='all',
                 num_layers=30, #smollm
                 num_heads=9, #smollm 
                 qk=True,
                 name='llama',
                 classifier=False,
                 add_lm_head=True):
        super().__init__(model)
        
        self.layer_name = layer_name
        self.enc_prefix = enc_prefix
        self.merge_type = merge_type
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.modules = modules
        self.qk = qk
        self.name = name
        self.classifier = classifier 
        self.add_lm_head = add_lm_head

    def add_layerblock_nodes(self, name_prefix, input_node, merge_type):
        # first half
        modules = self.modules

        emb_before_ln = input_node

        # LayerNorm
        input_node = self.add_nodes_from_sequence(name_prefix, [modules['emb_ln']], input_node)
        input_node = self.add_nodes_from_sequence(name_prefix, [NodeType.POSTFIX], input_node)

        # do attention block here
        residual = input_node

        # attention
        value_node = self.add_nodes_from_sequence(name_prefix, [modules['v']], residual) # NodeType.POSTFIX is being removed because P_{MHA} is shared by all attn weights anyway
        key_node = self.add_nodes_from_sequence(name_prefix, [modules['k'], NodeType.POSTFIX], residual)
        input_node = self.add_nodes_from_sequence(name_prefix, [modules['q'], NodeType.POSTFIX, NodeType.SUM], residual)
        self.add_directed_edge(key_node, input_node) # add key to "SUM" - it is really just a product but same handler
        input_node = self.add_nodes_from_sequence(name_prefix, [NodeType.SUM], input_node) #sum (mult)node to outproj
        self.add_directed_edge(value_node, input_node) #value node to sum (mult)

        # add self attn out proj to dot prod, layer norm, sum residual
        # get intermeds between attn and self attn out proj
        # get first residual vector from after self attn layer norm
        input_node = self.add_nodes_from_sequence(name_prefix, 
                                                [NodeType.PREFIX, modules['attn_o'], NodeType.SUM], 
                                                input_node) 
        self.add_directed_edge(emb_before_ln, input_node)

        normed_post_attn = self.add_nodes_from_sequence(name_prefix, [modules['post_attn_norm'], NodeType.POSTFIX], 
                                                  input_node=input_node)


        # Note rotary embeddings are skipped. Since they don't have weights, it seems the aren't required.
        # Not sure if this is correct. Consider adding Rotary back in.

        # MLP
        up_proj = self.add_nodes_from_sequence(name_prefix, [modules['up_proj']], normed_post_attn) # removed postfix
        gate_proj = self.add_nodes_from_sequence(name_prefix, [modules['gate_proj'], NodeType.SUM], normed_post_attn) # removed postfix
        self.add_directed_edge(up_proj, gate_proj)
        # removed Silu and use prefix instead of postfix
        mlp_down = self.add_nodes_from_sequence(name_prefix, [NodeType.PREFIX, modules['down_proj'],  NodeType.SUM], input_node=gate_proj) 
        self.add_directed_edge(input_node, mlp_down)

        return mlp_down

    def add_layer_nodes(self, layer_prefix, input_node, merge_type):
        source_node = input_node
        
        for layer_index in range(self.num_layers): # for graph visualization
        #for layer_index, layerblock in enumerate(self.get_module(name_prefix)):
            source_node = self.add_layerblock_nodes(f'{self.enc_prefix}.{layer_prefix}.{layer_index}', source_node, merge_type)        
        return source_node

    def graphify(self):
        modules = self.modules
        # keep input node
        input_node = self.create_node(node_type=NodeType.INPUT)
        # input_node -> emb_tok 
        emb_name = modules['emb']
        emb_node = self.create_node(node_type=NodeType.EMBEDDING, 
                                    layer_name=f'{self.enc_prefix}.{emb_name}'.strip('.'),
                                    param_name=f'{self.enc_prefix}.{emb_name}.weight'.strip('.'))
        self.add_directed_edge(input_node, emb_node)

        # after embedding, repeat LlamaDecoderLayer
        input_node = self.add_layer_nodes(f'{self.layer_name}', emb_node, self.merge_type)
        if self.add_lm_head:
            input_node = self.add_nodes_from_sequence(self.enc_prefix, [modules['final_norm'], NodeType.POSTFIX], input_node)

            input_node = self.add_nodes_from_sequence("", [modules['lm_head']], input_node)
            output_node = self.create_node(node_type=NodeType.OUTPUT)
            self.add_directed_edge(input_node, output_node)
        else:
            input_node = self.add_nodes_from_sequence(self.enc_prefix, [modules['final_norm'], NodeType.POSTFIX], input_node)
        
        return self
        
        '''
        BERT stuff
        # xformer layers -> dense -> layernorm -> output
        if self.name == 'bert' and self.classifier == False:
            dense_node = self.add_nodes_from_sequence(modules['head_pref'], ['transform.dense', 'transform.LayerNorm', NodeType.PREFIX, 'decoder'], input_node)
            output_node = self.create_node(node_type=NodeType.OUTPUT)
            self.add_directed_edge(dense_node, output_node)
        elif self.name == 'bert' and self.classifier == True:
            pool_node = self.add_nodes_from_sequence(self.enc_prefix, [modules['pooler']], input_node)
            class_node = self.add_nodes_from_sequence('', [NodeType.PREFIX, modules['classifier']], pool_node)
            output_node = self.create_node(node_type=NodeType.OUTPUT)
            self.add_directed_edge(class_node, output_node)
        elif self.name == 'roberta':
            #dense_node = self.add_nodes_from_sequence(modules['head_pref'], ['dense', NodeType.PREFIX, 'out_proj'], input_node)
            output_node = self.create_node(node_type=NodeType.OUTPUT)
            self.add_directed_edge(input_node, output_node)       
        
        return self
        '''

    
def bert(model, merge_type='ff_only', qk=False, classifier=False):
    modules = {'emb': 'embeddings.word_embeddings',
     'emb_pos': 'embeddings.position_embeddings',
     'emb_tok_type': 'embeddings.token_type_embeddings',
     'emb_ln': 'embeddings.LayerNorm',
     'q': 'attention.self.query',
     'k': 'attention.self.key',
     'v': 'attention.self.value',
     'lin_attn': 'attention.output.dense',
     'attn_ln': 'attention.output.LayerNorm',
     'fc1': 'intermediate.dense',
     'fc2': 'output.dense',
     'final_ln': 'output.LayerNorm',
     'head_pref': 'cls.predictions',
     'pooler': 'pooler.dense',
     'classifier': 'classifier'}
    return TransformerEncoderGraph(model, 
                                   modules,
                                   layer_name='bert.encoder.layer', 
                                   enc_prefix='bert',
                                   merge_type=merge_type,
                                   num_layers=12,
                                   num_heads=12,
                                   qk=qk,
                                   name='bert',
                                   classifier=classifier)



'''
checks if two state_dicts are the same. Used for debugging purposes.
reference: https://gist.github.com/rohan-varma/a0a75e9a0fbe9ccc7420b04bff4a7212 
'''
def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda":
            v_2 = v_2.to("cuda" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2, atol=1e-03):
            print(k_1)
            print(f"Tensor mismatch: {v_1} vs {v_2}")


if __name__ == '__main__':
    # unit test, nice
    # call from root directory with `python -m "graphs.resnet_graph"`
 

    # todo: change this to a text input
    data_x = torch.rand(4, 3, 224, 224)
    # todo: revisit this because it's not a classification task
    data_y = torch.zeros(4)

    dataset = TensorDataset(data_x, data_y)
    dataloader = DataLoader(dataset, batch_size=4)

    # this is a model
    model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    # not used yet
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    state_dict = model.state_dict()

    print(model)

    model3 = AutoModel.from_pretrained(model_name)
    model3.eval()

    # defaults merge the minimum so we need to add flags or change defaults
    # example = qk=True, merge_type='all', classifier=??
    graph1 = TransformerEncoderGraph(deepcopy(model)).graphify()
    graph2 = TransformerEncoderGraph(deepcopy(model)).graphify()

    # creates the merge model object and "adds hooks". Need to check if we need to re-impement this.
    merge = ModelMerge(graph1, graph2)

    # probably need to fix all the places that use .eval()
    # Does actual merging and writes to new state_dict to the model passed in.
    merge.transform(model3, dataloader, transform_fn=match_tensors_permute)

    graph1.draw(nodes=range(20), save_path='/tmp/graph1.png')
    graph1.draw(nodes=range(len(graph1.G)-20, len(graph1.G)), save_path='/tmp/graph2.png')

    print(model(data_x))

    # "merge" is a model so it should be able to do a forward pass but how does it know how to process
    # the input?
    print(merge(data_x))