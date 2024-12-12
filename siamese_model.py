
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import get_peft_model, PeftModel
import torch
import torch.nn as nn

def print_trainable_parameters(model):
    """
    print tuned parameter count
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


class SiameseModel():
    def __init__(self, model_path, device, data_device, peft_config=None, peft_path=None, mode=0, gate_rank=None, linear_A_path=None,
                  linear_B_path=None, top_rank=None, linear_top_A_path=None, linear_top_B_path=None,dropout_prob=0.0):

        self.fixed_model = None
        if "llama" in model_path.lower() or "qwen" in model_path.lower():
            self.fixed_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map={'': device})
        elif "chatglm" in model_path.lower():
            self.fixed_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map={'': device})
        self.fixed_model.half()
        self.fixed_model.eval()

        self.tuned_model = None
        if "llama" in model_path.lower() or "qwen" in model_path.lower():
            self.tuned_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map={'': device})
        elif "chatglm" in model_path.lower():
            self.tuned_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map={'': device})
        self.tuned_model.half()

        if peft_path is not None:
            self.tuned_model = PeftModel.from_pretrained(self.tuned_model, peft_path).to(device)

        if peft_config is not None:
            self.tuned_model = get_peft_model(self.tuned_model, peft_config).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.device = device
        self.data_device = data_device
        self.model_path = model_path

        for param in self.fixed_model.parameters():
            param.requires_grad = False

        print("fixed_model:")
        print_trainable_parameters(self.fixed_model)
        print("tuned_model:")
        print_trainable_parameters(self.tuned_model)

        # 获取hidden state的维度
        hidden_size = self.tuned_model.config.hidden_size
        self.mode = mode
        self.dropout_prob = dropout_prob
        # mode=0: Add equal proportions
        # mode=2: gate
        # mode=3: gate concat linear
        # mode=4: Residual
        # mode=5: Query-aware

        def load_or_initialize(linear_path_A, linear_path_B, in_features_A, out_features_A, out_features_B):
            if linear_path_A is not None and linear_path_B is not None:
                linear_A = torch.load(linear_path_A).to(self.data_device)
                linear_B = torch.load(linear_path_B).to(self.data_device)
            else:
                linear_A = nn.Sequential(
                    nn.Linear(in_features=in_features_A, out_features=out_features_A, bias=False).to(self.data_device),
                    nn.Dropout(self.dropout_prob)
                )
                linear_B = nn.Sequential(
                    nn.Linear(in_features=out_features_A, out_features=out_features_B, bias=False).to(self.data_device),
                    nn.Dropout(self.dropout_prob)
                )
            return linear_A, linear_B
        if self.mode == 2 or self.mode == 4 or mode==6:
            self.linear_A, self.linear_B = load_or_initialize(linear_A_path, linear_B_path, hidden_size * 2, gate_rank, hidden_size)
        elif self.mode == 5:
            self.linear_A, self.linear_B = load_or_initialize(linear_A_path, linear_B_path, hidden_size * 4, gate_rank, hidden_size)
        elif mode == 3:
            self.linear_A, self.linear_B = self.load_or_initialize(linear_A_path, linear_B_path, hidden_size * 2, gate_rank, hidden_size)
            self.linear_top_A, self.linear_top_B = self.load_or_initialize(linear_top_A_path, linear_top_B_path, hidden_size * 2, top_rank, hidden_size)

    def cal_logits(self, input_ids, attention_mask):

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # train_model logits
        train_model_logits = self.tuned_model(input_ids).logits  # torch.Size([1, 58, 65024])
        # fix_model logits
        fix_model_logits = self.fixed_model(input_ids).logits  # torch.Size([1, 58, 65024])

        logits = 0.5*train_model_logits + 0.5*fix_model_logits
        logits = logits[:, :-1, :]

        padding = torch.zeros(logits.size(0), 1, logits.size(2)).to(self.device)
        logits = torch.cat((padding, logits), dim=1)

        return logits


    def cal_logits_from_hidden_state_with_gate(self, input_ids, query_length=None):
        assert not torch.isnan(input_ids).any(), "input_ids contains nan"
 
        assert not torch.isnan(query_length).any(), "query_length contains nan"

        input_ids = input_ids.to(self.device)

        train_model_outputs = self.tuned_model(input_ids, output_hidden_states=True)
        if "llama" in self.model_path.lower() or "qwen" in self.model_path.lower():
            train_model_last_hidden_state = train_model_outputs.hidden_states[-1].to(self.data_device)
        elif "chatglm" in self.model_path.lower():
            train_model_last_hidden_state = train_model_outputs.hidden_states[-1].permute(1, 0, 2)
            train_model_last_hidden_state = self.tuned_model.transformer.encoder.final_layernorm(
                train_model_last_hidden_state).to(self.data_device)

        fix_model_outputs = self.fixed_model(input_ids, output_hidden_states=True)
        if "llama" in self.model_path.lower() or "qwen" in self.model_path.lower():
            fix_model_last_hidden_state = fix_model_outputs.hidden_states[-1].to(self.data_device)
        elif "chatglm" in self.model_path.lower():
            fix_model_last_hidden_state = fix_model_outputs.hidden_states[-1].permute(1, 0, 2)
            fix_model_last_hidden_state = self.fixed_model.transformer.encoder.final_layernorm(
                fix_model_last_hidden_state).to(self.data_device)

        last_hidden_state = None
        if self.mode in [2,3,4]:
            concat_hidden_state = torch.cat((fix_model_last_hidden_state, train_model_last_hidden_state), dim=2).float()
            compressed_hidden_state = self.linear_B(self.linear_A(concat_hidden_state))
            gate = torch.sigmoid(compressed_hidden_state)

            last_hidden_state = gate * fix_model_last_hidden_state + (1 - gate) * train_model_last_hidden_state

        elif self.mode == 5:
            query_length = query_length.to(self.data_device)

            batch_size, seq_len, hidden_size = train_model_last_hidden_state.size()
            #print("batch_size")
            #print(batch_size)
            #print("seq_len")
            #print(seq_len)
            #print("hidden_size")
            #print(hidden_size)

            mask = torch.arange(seq_len).expand(batch_size, seq_len).to(
                train_model_last_hidden_state.device) < query_length.unsqueeze(1)
            #print("mask")
            #print(mask)
            #print(train_model_last_hidden_state.size())
            #print(mask.size())
            #print(mask.unsqueeze(2).size())

            train_model_masked_hidden_state = train_model_last_hidden_state * mask.unsqueeze(2)
            train_model_query_avg_hidden_states = train_model_masked_hidden_state.sum(dim=1) / mask.sum(dim=1, keepdim=True)

            fix_model_masked_hidden_state = fix_model_last_hidden_state * mask.unsqueeze(2)
            fix_model_query_avg_hidden_states = fix_model_masked_hidden_state.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            
            train_model_query_avg_hidden_states = train_model_query_avg_hidden_states.unsqueeze(1).expand(-1, seq_len, -1)
            fix_model_query_avg_hidden_states = fix_model_query_avg_hidden_states.unsqueeze(1).expand(-1, seq_len, -1)
            #print(fix_model_query_avg_hidden_states.size())
            #print(fix_model_last_hidden_state.size())
            #print(train_model_query_avg_hidden_states.size())
            #print(train_model_last_hidden_state.size())
            concat_hidden_state = torch.cat((fix_model_query_avg_hidden_states,fix_model_last_hidden_state, train_model_last_hidden_state,train_model_query_avg_hidden_states), dim=2).float()
            compressed_hidden_state = self.linear_B(self.linear_A(concat_hidden_state))
            gate = torch.sigmoid(compressed_hidden_state)

            last_hidden_state = gate * fix_model_last_hidden_state + (1 - gate) * train_model_last_hidden_state

        elif self.mode == 6:
            query_length = query_length.to(self.data_device)
            batch_size, seq_len, hidden_size = train_model_last_hidden_state.size()
            mask = torch.arange(seq_len).expand(batch_size, seq_len).to(
                train_model_last_hidden_state.device) < query_length.unsqueeze(1)
            train_model_masked_hidden_state = train_model_last_hidden_state * mask.unsqueeze(2)
            train_model_query_avg_hidden_states = train_model_masked_hidden_state.sum(dim=1) / mask.sum(dim=1,
                                                                                                        keepdim=True)

            fix_model_masked_hidden_state = fix_model_last_hidden_state * mask.unsqueeze(2)
            fix_model_query_avg_hidden_states = fix_model_masked_hidden_state.sum(dim=1) / mask.sum(dim=1, keepdim=True)
          
            fix_model_query_avg_hidden_states = fix_model_query_avg_hidden_states.unsqueeze(1).expand(-1, fix_model_last_hidden_state.size(1), -1)
            train_model_query_avg_hidden_states = train_model_query_avg_hidden_states.unsqueeze(1).expand(-1, train_model_last_hidden_state.size(1), -1)

            fix_model_last_hidden_state = 0.5*fix_model_last_hidden_state + 0.5*fix_model_query_avg_hidden_states
            train_model_last_hidden_state = 0.5*train_model_last_hidden_state +0.5*train_model_query_avg_hidden_states
            concat_hidden_state = torch.cat((fix_model_last_hidden_state, train_model_last_hidden_state), dim=2).float()
            compressed_hidden_state = self.linear_B(self.linear_A(concat_hidden_state))
            gate = torch.sigmoid(compressed_hidden_state)

            last_hidden_state = gate * fix_model_last_hidden_state + (1 - gate) * train_model_last_hidden_state

        logits=None
        if self.mode == 2 or self.mode == 5 or self.mode==6:
            last_hidden_state = last_hidden_state.to(self.device).half()
            if "llama" in self.model_path.lower() or "qwen" in self.model_path.lower():
                logits = self.fixed_model.lm_head(last_hidden_state)
            elif "chatglm" in self.model_path.lower():
                logits = self.fixed_model.transformer.output_layer(last_hidden_state)

        elif self.mode==3:
            new_hidden_state = self.linear_top_B(self.linear_top_A(torch.cat((fix_model_last_hidden_state, last_hidden_state), dim=2))).to(self.device).half()
            if "llama" in self.model_path.lower() or "qwen" in self.model_path.lower():
                logits = self.fixed_model.lm_head(new_hidden_state)
            elif "chatglm" in self.model_path.lower():
                logits = self.fixed_model.transformer.output_layer(new_hidden_state)

        elif self.mode == 4:
            new_hidden_state = (0.5 * fix_model_last_hidden_state + 0.5 * last_hidden_state).half().to(self.device)
            if "llama" in self.model_path.lower() or "qwen" in self.model_path.lower():
                logits = self.fixed_model.lm_head(new_hidden_state)
            elif "chatglm" in self.model_path.lower():
                logits = self.fixed_model.transformer.output_layer(new_hidden_state)

        logits = logits[:, :-1, :]
        padding = torch.zeros(logits.size(0), 1, logits.size(2)).to(self.device)
        logits = torch.cat((padding, logits), dim=1)

        return logits

    def decode_next_token(self, logits):
        next_token_id = torch.argmax(logits, dim=-1).to(self.device)
        return next_token_id

    def forward(self, input_text, max_target_len):
        output_token_ids = []

        inputs = self.tokenizer(text=input_text, add_special_tokens=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)  # Convert to tensor and add batch dimension
        attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(self.device)

        for i in range(max_target_len):
            with torch.no_grad():
                fix_model_logits = self.fixed_model(input_ids).logits[:, -1, :]  # torch.Size([1, 58, 65024])
                train_model_logits = self.tuned_model(input_ids).logits[:, -1, :]  # torch.Size([1, 58, 65024])
                next_logits = 0.5*fix_model_logits + 0.5*train_model_logits

            next_token_id = self.decode_next_token(next_logits)
            output_token_ids.append(next_token_id.item())  # Append the integer representation of token ID


            if next_token_id == self.tokenizer.eos_token_id:
                break
            else:
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)  # Concatenate new token
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id).unsqueeze(0)], dim=1)
        output = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)

        return output

    def forward_from_hidden_state_with_gate(self, input_text, max_target_len):
        output_token_ids = []

        input_ids = self.tokenizer.encode(text=input_text, add_special_tokens=True)
        seq_len = len(input_ids)
        batch_size = 1
        query_length = torch.tensor(len(input_ids)).unsqueeze(0).to(self.data_device)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)  # Convert to tensor and add batch dimension
        cal_query_hidden_state = True

        for i in range(max_target_len):
            with torch.no_grad():
                train_model_outputs = self.tuned_model(input_ids, output_hidden_states=True)
                train_model_last_hidden_state = None
                if "llama" in self.model_path.lower() or "qwen" in self.model_path.lower():
                    train_model_last_hidden_state = train_model_outputs.hidden_states[-1].to(self.data_device)  # [1,seq,2560] 最后一个hidden_state生成下一个token
                elif "chatglm" in self.model_path.lower():
                    train_model_last_hidden_state = train_model_outputs.hidden_states[-1].permute(1, 0, 2)
                    train_model_last_hidden_state = self.fixed_model.transformer.encoder.final_layernorm(
                        train_model_last_hidden_state).to(self.data_device)

                fix_model_outputs = self.fixed_model(input_ids, output_hidden_states=True)
                fix_model_last_hidden_state = None
                if "llama" in self.model_path.lower() or "qwen" in self.model_path.lower():
                    fix_model_last_hidden_state = fix_model_outputs.hidden_states[-1].to(self.data_device)
                elif "chatglm" in self.model_path.lower():
                    fix_model_last_hidden_state = fix_model_outputs.hidden_states[-1].permute(1, 0, 2)
                    fix_model_last_hidden_state = self.fixed_model.transformer.encoder.final_layernorm(fix_model_last_hidden_state).to(self.data_device)

                next_token_fix_model_last_hidden_state = fix_model_last_hidden_state[:, -1, :]
                next_token_train_model_last_hidden_state = train_model_last_hidden_state[:, -1, :]

                last_hidden_state = None
                if (self.mode == 5 or self.mode==6) and cal_query_hidden_state:
                    mask = torch.arange(seq_len).expand(batch_size, seq_len).to(
                        self.data_device) < query_length.unsqueeze(1)
                    train_model_masked_hidden_state = train_model_last_hidden_state * mask.unsqueeze(2)
                    train_model_query_avg_hidden_states = train_model_masked_hidden_state.sum(dim=1) / mask.sum(dim=1,keepdim=True)

                    fix_model_masked_hidden_state = fix_model_last_hidden_state * mask.unsqueeze(2)
                    fix_model_query_avg_hidden_states = fix_model_masked_hidden_state.sum(dim=1) / mask.sum(dim=1,keepdim=True)
                    cal_query_hidden_state = False

                if self.mode in [2,3,4]:
                    concat_hidden_state = torch.cat((next_token_fix_model_last_hidden_state, next_token_train_model_last_hidden_state),dim=-1).float()
                    compressed_hidden_state = self.linear_B(self.linear_A(concat_hidden_state))

                    gate = torch.sigmoid(compressed_hidden_state)
                    last_hidden_state = gate * next_token_fix_model_last_hidden_state + (1 - gate) * next_token_train_model_last_hidden_state

                if self.mode == 5:

                    concat_hidden_state = torch.cat((fix_model_query_avg_hidden_states, next_token_fix_model_last_hidden_state,
                                                     next_token_train_model_last_hidden_state,
                                                     train_model_query_avg_hidden_states), dim=-1).float()
                    compressed_hidden_state = self.linear_B(self.linear_A(concat_hidden_state))
                    gate = torch.sigmoid(compressed_hidden_state)

                    last_hidden_state = gate * next_token_fix_model_last_hidden_state + (1 - gate) * next_token_train_model_last_hidden_state
                    #print(last_hidden_state.size())

                if self.mode == 6:
                    next_token_fix_model_last_hidden_state = 0.5*next_token_fix_model_last_hidden_state+0.5*fix_model_query_avg_hidden_states
                    next_token_train_model_last_hidden_state = 0.5*next_token_train_model_last_hidden_state + 0.5*train_model_query_avg_hidden_states

                    concat_hidden_state = torch.cat((next_token_fix_model_last_hidden_state,
                                                     next_token_train_model_last_hidden_state,
                                                     ), dim=-1).float()
                    compressed_hidden_state = self.linear_B(self.linear_A(concat_hidden_state))
                    gate = torch.sigmoid(compressed_hidden_state)

                    last_hidden_state = gate * next_token_fix_model_last_hidden_state + (1 - gate) * next_token_train_model_last_hidden_state
                    #print(last_hidden_state.size())


                next_logits = None
                if self.mode==2 or self.mode==5 or self.mode==6:
                    last_hidden_state = last_hidden_state.half().to(self.device)
                    if "llama" in self.model_path.lower() or "qwen" in self.model_path.lower(): 
                        next_logits = self.fixed_model.lm_head(last_hidden_state)
                    elif "chatglm" in self.model_path.lower():
                        next_logits = self.fixed_model.transformer.output_layer(last_hidden_state)

                if self.mode==3:
                    new_hidden_state = self.linear_top_B(self.linear_top_A(torch.cat((next_token_fix_model_last_hidden_state, last_hidden_state), dim=-1))).half().to(self.device)
                    if "llama" in self.model_path.lower() or "qwen" in self.model_path.lower(): 
                        next_logits = self.fixed_model.lm_head(new_hidden_state)
                    elif "chatglm" in self.model_path.lower():
                        next_logits = self.fixed_model.transformer.output_layer(new_hidden_state)

                if self.mode == 4:
                    new_hidden_state = (0.5 * next_token_fix_model_last_hidden_state + 0.5 * last_hidden_state).half().to(self.device)
                    if "llama" in self.model_path.lower() or "qwen" in self.model_path.lower(): 
                        next_logits = self.fixed_model.lm_head(new_hidden_state)
                    elif "chatglm" in self.model_path.lower():
                        next_logits = self.fixed_model.transformer.output_layer(new_hidden_state)



            next_token_id = self.decode_next_token(next_logits)
            #print(next_token_id)
            output_token_ids.append(next_token_id.item())  # Append the integer representation of token ID

            if next_token_id == self.tokenizer.eos_token_id:
                break
            else:
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)  # Concatenate new token
                seq_len = len(input_ids)
        output = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)

        return output

    def count_trainable_parameters(self):
        params = sum(p.numel() for p in self.linear_A.parameters() if p.requires_grad) + sum(
            p.numel() for p in self.linear_B.parameters() if p.requires_grad)
        if self.mode == 3:
            params += sum(p.numel() for p in self.linear_top_A.parameters() if p.requires_grad) + sum(
                p.numel() for p in self.linear_top_B.parameters() if p.requires_grad)
        return params

    def set_requires_grad(self, model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def train(self):
        self.tuned_model.train()

        if self.mode in [2, 3,4,5]:
            self.linear_A.train()
            self.linear_B.train()
            self.set_requires_grad(self.linear_A, True)
            self.set_requires_grad(self.linear_B, True)
            if self.mode == 3:
                self.set_requires_grad(self.linear_top_A, True)
                self.set_requires_grad(self.linear_top_B, True)
                
            num_params_with_bias = self.count_trainable_parameters()
            print("linear parameter count:", num_params_with_bias)

        print_trainable_parameters(self.fixed_model)
        print_trainable_parameters(self.tuned_model)

    def eval(self):
        self.tuned_model.eval()

        if self.mode in [2, 3, 4,5]:
            self.linear_A.eval()
            self.linear_B.eval()
            self.set_requires_grad(self.linear_A, False)
            self.set_requires_grad(self.linear_B, False)
            if self.mode == 3:
                self.set_requires_grad(self.linear_top_A, False)
                self.set_requires_grad(self.linear_top_B, False)

            num_params_with_bias = self.count_trainable_parameters()
            print("linear parameter count:", num_params_with_bias)

        print_trainable_parameters(self.fixed_model)
        print_trainable_parameters(self.tuned_model)

