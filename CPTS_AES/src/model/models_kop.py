
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
import json

#################################
# ASAP++ 실험 모델
#################################
    
class AsapBaseRegression_v2(nn.Module):

    def __init__(self, config, pre_trained_model, num_labels, n_hidden):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.n_hidden = n_hidden
        self.pre_trained_model = pre_trained_model

        self.gru = nn.GRU(config.hidden_size, self.n_hidden, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.n_hidden * 2, self.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            am_label=None,
    ):

        if am_label is not None:
            discriminator_hidden_states = self.pre_trained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                am_label=am_label
            )
        else:
            discriminator_hidden_states = self.pre_trained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # last-layer hidden state
        # sequence_output: [batch_size, seq_length, hidden_size]
        sequence_output = discriminator_hidden_states[0]

        # sequence_output: [batch, seq_length, hidden*2]
        # hidden: [2, batch, hidden]
        sequence_output, hidden = self.gru(sequence_output)
        final_hidden = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1) 

        logits = torch.sigmoid(self.linear(final_hidden))  # [batch, num_labels]     # sigmoid 추가

        outputs = (logits,) + discriminator_hidden_states[2:]

        return outputs
    

class EssayEmbeddingModel(nn.Module):
    def __init__(self, model_name_or_path):
        super(EssayEmbeddingModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

        # hidden_size를 512로 줄이는 Linear 레이어 추가
        self.resize = nn.Linear(self.model.config.hidden_size, 512)

    def forward(self, text):
        # 텍스트 토큰화 (길이를 510으로 설정)
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512,
                                pad_to_max_length=True)

        # 임베딩 추출
        with torch.no_grad():
            outputs = self.model(**inputs)

        # last_hidden_state 차원 변경
        resized_output = self.resize(outputs['last_hidden_state'])
        return resized_output

class multihead_attention(nn.Module):
    def __init__(self, num_units, num_heads=1, dropout_rate=0, gpu=True, causality=False):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(multihead_attention, self).__init__()
        self.gpu = gpu
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())

        # Use GPU
        if self.gpu:
            self.Q_proj = self.Q_proj.cuda()
            self.K_proj = self.K_proj.cuda()
            self.V_proj = self.V_proj.cuda()

        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values, last_layer=False):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)

        # get dim to concat
        concat_dim = len(Q.shape) - 1

        if concat_dim == 1:
            Q = Q.unsqueeze(dim=1)
            queries = queries.unsqueeze(dim=1)
            concat_dim = 2

        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=concat_dim), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=concat_dim), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=concat_dim), dim=0)  # (h*N, T_q, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Activation
        if not last_layer:
            outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)

        outputs = outputs * query_masks

        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)

        if last_layer:
            if self.num_heads > 1:
                outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=concat_dim)
            return outputs

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=concat_dim)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        return outputs

class gateway_sw(nn.Module):#####유력후보

    # GRU의 마지막 state 값을 max-pooling 해서 모두 사용

    def __init__(self, config, pre_trained_model, num_labels, n_hidden):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.n_hidden = n_hidden
        self.pre_trained_model = pre_trained_model
    
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.essay_label_embedding = nn.Embedding(7, 768, padding_idx=0)

        self.attention1 = multihead_attention(num_units=768)
        self.attention2 = multihead_attention(num_units=768)
        self.attention3 = multihead_attention(num_units=768)
        self.attention4 = multihead_attention(num_units=768)
        self.attention5 = multihead_attention(num_units=768)
        self.attention6 = multihead_attention(num_units=768)
        self.attention7 = multihead_attention(num_units=768)
        self.attention8 = multihead_attention(num_units=768)
        self.attention9 = multihead_attention(num_units=768)
        self.attention10 = multihead_attention(num_units=768)
        self.attention11 = multihead_attention(num_units=768)

        self.gru = nn.GRU(config.hidden_size, self.n_hidden, bidirectional=True, batch_first=True)

        self.layer1 = nn.Linear(512,1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            am_label=None,
            essay_amlabel=None,
    ):

        if am_label is not None:
            discriminator_hidden_states = self.pre_trained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                #am_label=am_label
            )
        else:
            discriminator_hidden_states = self.pre_trained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # last-layer hidden state
        # sequence_output: [batch_size, seq_length, hidden_size]
        sequence_output = discriminator_hidden_states[0]
        am_embd = self.essay_label_embedding(essay_amlabel)

        #import pdb;pdb.set_trace()

        attention_output1 = self.attention1(am_embd, sequence_output, sequence_output) #b, s, h
        attention_output2 = self.attention2(am_embd, sequence_output, sequence_output) #b, s, h
        attention_output3 = self.attention3(am_embd, sequence_output, sequence_output) #b, s, h
        attention_output4 = self.attention4(am_embd, sequence_output, sequence_output) #b, s, h
        attention_output5 = self.attention5(am_embd, sequence_output, sequence_output) #b, s, h
        attention_output6 = self.attention6(am_embd, sequence_output, sequence_output) #b, s, h
        attention_output7 = self.attention7(am_embd, sequence_output, sequence_output) #b, s, h
        attention_output8 = self.attention8(am_embd, sequence_output, sequence_output) #b, s, h
        attention_output9 = self.attention9(am_embd, sequence_output, sequence_output) #b, s, h
        attention_output10 = self.attention10(am_embd, sequence_output, sequence_output) #b, s, h
        attention_output11 = self.attention11(am_embd, sequence_output, sequence_output) #b, s, h

        eloutput1 = attention_output1 + sequence_output
        eloutput2 = attention_output2 + sequence_output
        eloutput3 = attention_output3 + sequence_output
        eloutput4 = attention_output4 + sequence_output
        eloutput5 = attention_output5 + sequence_output
        eloutput6 = attention_output6 + sequence_output
        eloutput7 = attention_output7 + sequence_output
        eloutput8 = attention_output8 + sequence_output
        eloutput9 = attention_output9 + sequence_output
        eloutput10 = attention_output10 + sequence_output
        eloutput11 = attention_output11 + sequence_output

        sum_tensors = torch.cat([eloutput1,eloutput2,eloutput3,eloutput4,eloutput5,eloutput6,eloutput7,eloutput8,eloutput9, eloutput10, eloutput11], dim=0)
        sequence_output, hidden1 = self.gru(sum_tensors)
        final_hidden1 = torch.cat([hidden1[0,:,:], hidden1[1,:,:]], dim=-1) # [9*b, h]
        total_elements = final_hidden1.nelement()
        batch_size = total_elements // (11 *self.n_hidden*2)

        try:
            reshaped_tensor = final_hidden1.reshape(11, batch_size, self.n_hidden*2).transpose(0, 1) #[9, 8(batch), h] ==> (batch, n_trait, hidden)
        except:
            print("hi")
        logits = torch.sigmoid(self.layer1(reshaped_tensor)).squeeze(-1)  # [8(batch), n_trait, 1] ==> (batch, n_trait)

        outputs = (logits,) + discriminator_hidden_states[2:]

        return outputs