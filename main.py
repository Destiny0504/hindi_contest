import pandas as pd
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset
from torch.utils.tensorboard import SummaryWriter

#  *** maybe it can merge with tamil moel ***


class hindi_model():
    def __init__(self,model_name, hindi_data, tokenized_len = 500, hidden_state = 100):
        # let data store by differet title
        self.model_name = model_name
        self.context = list(hindi_data['context'])
        self.question = list(hindi_data['question'])
        self.answer = list(hindi_data['answer_text'])
        self.answer_start = list(hindi_data['answer_start'])

        config = transformers.AutoConfig.from_pretrained(self.model_name)

        # ***this is a better way to update your config*** follow this way on other project
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": 0.1,  # hyperparameter
                "layer_norm_eps": 1e-7,  # hyperparameter
                "add_pooling_layer": False,
                "num_labels": 3,
            }
        )
        # fixed random seed (let others can do the same expriment as yours)
        torch.manual_seed(1428)

        # loading model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = transformers.AutoModel.from_pretrained(self.model_name)
        print(f'config.num_labels:{config.num_labels}')
        self.output = torch.nn.Linear(config.hidden_size, config.num_labels)
        #self.output_2 = torch.nn.Linear(hidden_state, config.num_labels)

        # each_len is use for set each slice's context length
        self.each_len = tokenized_len

        # get answer length (can get the answer by self.context[0][ self.start_txt[0] : self.start_txt[0] + self.answer_len[0]])
        self.answer_len = self.answer_text_len()

        # create dataset for training
        self.dataset = self.create_dataset()
        self.dataloader = self.change_dataset_into_dataloader()

    # tokenized each context and concatenate with its question  (also use for question) special token ID:2: [PAD] 3: [CLS] 4: [SEP]

    # 2 is for test it need to be changed
    def create_dataset(self, total_context = 700, context_token_max_length = 200, question_token_max_length = 50):
        #print(self.tokenizer.convert_ids_to_tokens([3, 8943, 3429, 7639, 6540, 36435, 3054, 2, 4]))
        tokenized_data = []
        finished_dataset = []
        answer_label = []
        all_correct = 0
        for i in range(total_context):
            tokenized_question = self.tokenizer(self.question[i], add_special_tokens = True, max_length = question_token_max_length, truncation = True, padding = 'max_length')
            tmp = self.split_context_into_smaller_size(self.context[i])
            for j in range(len(tmp)):
            
                if self.answer[i] in tmp[j] and (self.answer_start[i] < (j + 1) * self.each_len and self.answer_start[i] > j * self.each_len):
                    # this part is concatenating the inputs
                    tokenized_context = self.tokenizer(tmp[j], add_special_tokens = True, max_length = context_token_max_length, truncation = True, padding='max_length')
                
                    model_input_ids = tokenized_context.input_ids + tokenized_question.input_ids
                    model_attention_mask = tokenized_context.attention_mask + tokenized_question.attention_mask
                    #print(tokenized_context.token_type_ids)
                    
                    for itr in range(len(tokenized_question.token_type_ids)):
                        tokenized_question.token_type_ids[itr] = 1
                    #print(tokenized_question.token_type_ids)
                    model_token_type_ids = tokenized_context.token_type_ids + tokenized_question.token_type_ids

                    tokenized_answer = self.tokenizer(self.answer[i], add_special_tokens = True)

                    print(f'anwer before token:{self.answer[i]}\ntokenized answer:{tokenized_answer}\ntoken to id{self.tokenizer.convert_ids_to_tokens(tokenized_answer.input_ids)}')

                    for itr in range(context_token_max_length - len(tokenized_answer.input_ids)):
                        #print(self.tokenizer.convert_ids_to_tokens(tokenized_context.input_ids[itr: itr + len(tokenized_answer.input_ids) - 2]))  # the reason why we need to minus 2 is tokenized_answer.input_ids has 3 in the front and 4 in the end 
                        if tokenized_context.input_ids[itr : itr + len(tokenized_answer.input_ids) - 2] == tokenized_answer.input_ids[1 : -1]:                          
                            answer_label = [itr, itr + len(tokenized_answer.input_ids) - 2]                    
                            print("find\n")
                            all_correct = all_correct + 1

                    # this part is creating the correct label in the untokenized context ( a better solution, but we want to start training first )
                    start_positon = tmp[j].find(self.answer[i])

                    #answer_label = [start_positon, (start_positon + self.answer_len[i] - 1)]

                    # put the data into the list
                    if len(answer_label) != 0:
                        tokenized_data.append({'input_ids': model_input_ids, 'attention_mask': model_attention_mask, 'token_type_ids': model_token_type_ids, 'label': answer_label})
                    else:
                        answer_label = [-1, -1]
                        tokenized_data.append({'input_ids': model_input_ids, 'attention_mask': model_attention_mask, 'token_type_ids': model_token_type_ids, 'label': answer_label})

                else:
                    # this part is concatenating the inputs
                    tokenized_context = self.tokenizer(tmp[j], add_special_tokens = True, max_length = context_token_max_length, truncation = True, padding = 'max_length')
                    model_input_ids = tokenized_context.input_ids + tokenized_question.input_ids
                    model_attention_mask = tokenized_context.attention_mask + tokenized_question.attention_mask
                    #print(tokenized_context.token_type_ids)
                    for itr in range(len(tokenized_question.token_type_ids)):
                        tokenized_question.token_type_ids[itr] = 1
                    #print(tokenized_question.token_type_ids)
                    model_token_type_ids = tokenized_context.token_type_ids + tokenized_question.token_type_ids

                    # this part is creating the correct label
                    answer_label = [-1, -1]
                    # put the data into the list
                    tokenized_data.append({'input_ids': model_input_ids, 'attention_mask': model_attention_mask, 'token_type_ids': model_token_type_ids, 'label': answer_label})
        
        print(all_correct)
        input_ids = []
        attention_mask = []
        token_type_ids = []
        label = []

        for i in range(len(tokenized_data)):
            # print(tokenized_data[i])
            input_ids.append(tokenized_data[i]['input_ids'])
            attention_mask.append(tokenized_data[i]['attention_mask'])
            token_type_ids.append(tokenized_data[i]['token_type_ids'])
            label.append(tokenized_data[i]['label'])

        finished_dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(input_ids),
            torch.LongTensor(attention_mask),
            torch.LongTensor(token_type_ids),
            torch.LongTensor(label)
        )

        return finished_dataset

    def change_dataset_into_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            dataset = self.dataset,
            batch_size = 8,
            shuffle = True
        )
        return dataloader

    # change the context into a smaller size to throw into model and tokenized
    def split_context_into_smaller_size(self, data):
        splited_context = []
        tmp = int(len(data) / self.each_len)
        if tmp > 0:
            for i in range(tmp):
                splited_context.append(
                    data[self.each_len * i: self.each_len * (i + 1)])
            splited_context.append(data[self.each_len * (i + 1):])
        else:
            splited_context.append(data[0:])
        return splited_context

    def answer_text_len(self):
        tmp = []
        for i in range(len(self.answer)):
            tmp.append(len(self.answer[i]))
        return tmp

    def setting_optimizer_parameter(self):
        weight_decay = 0.01
        learning_rate = 1e-4
        adam_epsilon = 1e-6
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr = self.learning_rate, eps = adam_epsilon)

    def test_transformer(self):
        loss = 0.0
        epoch_iterator = tqdm(self.dataloader, desc=f'loss{loss}')
        for step, batch in enumerate(epoch_iterator):
            output = self.model(input_ids = batch[0], attention_mask = batch[1], token_type_ids = batch[2])
            print('===================')
            #print(f'step{step}')
            #print(f'batch{batch}')
            print(f'output : {output[0]}')
            print(len(output[0][0]))
            for i in range(len(output[0])):
                for j in range(len(output[0][i])):
                    print(f'output[0][i][j] : {output[0][i][j]}')
                    tmp = self.output(output[0][i][j])
                    print(tmp) 

            logits = self.output(output[0])
            #print(logits)
            B_logits, I_logits, O_logits = logits.split(1, dim = -1) ###
            print(f'B logits:{B_logits}')
            print(f'I logits:{I_logits}')
            print(f'O logits:{O_logits}')
            # transport a n * 1 vector into a 1 * n vector
            B_logits = B_logits.squeeze(-1).contiguous() 
            I_logits = I_logits.squeeze(-1).contiguous()
            O_logits = O_logits.squeeze(-1).contiguous()
            #print(f'start logits:{start_logits}')
            #print(f'end logits:{end_logits}')
            print(len(B_logits))
            print(f'length of B_logits[0] : {len(B_logits[0])}')
            loss = self.loss_function(B_logits, I_logits, O_logits, batch[3])
            print('===================')

        return 'finished'
    
    def loss_function(self, B_logits, I_logits, O_logits, label): # the last step of the project
        print(label)
        batch_size = len(label[0])
        print(len(B_logits[0]))
        #print(f'start logits[0] length : {len(start_logits[0][0])}')
        for i in range(batch_size):
            if label[0][i] != -1 and label[0][i] != -1:
                B_label = torch.zeros(len(B_logits[i]))
                I_label = torch.zeros(len(I_logits[i]))
                O_label = torch.zeros(len(O_logits[i]))
                B_label[label[0]] = 1

                # change the label into the training form
                if label[0] != (label[1] - 1):
                    I_label[label[0] : label[1]] = 1
                for itr in range(len(O_label)):
                    if I_label[itr] == 0 and B_label[itr] == 0:
                        O_label[itr] = 1
     
                loss_fun = torch.nn.CrossEntropyLoss()
                loss = loss_fun(B_logits, B_label) + loss_fun(I_logits, I_label) + loss_fun(O_logits, O_label)
                
                print(loss)
                return loss
            else:
                return None

    def forword(self):
        loss = 0.0
        log_step = 50
        log_path = './exp/test_tmp'
        writer = SummaryWriter(log_path)

        self.model.zero_grad()
        self.model.train()

        for epoch in range(2):
            epoch_iterator = tqdm(self.dataloader, desc = f'loss{loss}')
            for step, batch in enumerate(epoch_iterator):
                output = model(
                    input_ids = batch[0],
                    attention_mask = batch[1],
                    token_type_ids = batch[2],
                    labels = batch[3]
                )

                loss = output.loss

                loss.backward()
                optimizer.step()
                model.zero_grad()

                if step % log_step == 0:
                    writer.add_scalar(f'loss', loss, step)
                    epoch_iterator.set_description(
                        f'epoch: {epoch}, loss: {loss:.6f}')
            writer.close()

# this class will not be used (maybe) because of adding another parameter to the hindi model
class tamil_model():
    def __init__(self, tamil_data):
        self.data = hindi_data
        self.model = transformers.AutoModel.from_pretrained("monsoon-nlp/tamillion")
        self.tokenizer = AutoTokenizer.from_pretrained("monsoon-nlp/tamillion")


def main():
    tamil_data, hindi_data = data_slice()
    tamil_data_question = tamil_data[['question']]
    hindi_data_question = hindi_data[['question']]
    model1 = hindi_model(model_name = "monsoon-nlp/hindi-bert",hindi_data = hindi_data)
    # print(model1.context[0][ model1.start_txt[0] : model1.start_txt[0] + model1.answer_len[0]]) test pass
    print(model1.test_transformer())


def data_split(data):
    data_train, data_test = train_test_split(data, test_size = 0.2)
    return data_train, data_test


def data_slice():
    df = pd.read_csv('./train.csv')

    df1 = df[df['language'].apply(lambda language: 'tamil' == language)]
    df2 = df[df['language'].apply(lambda language: 'hindi' == language)]

    # reset index of df1 and df2
    df1 = df1[['id', 'context', 'question', 'answer_text','answer_start', 'language']].reset_index(drop=True)
    df2 = df2[['id', 'context', 'question', 'answer_text','answer_start', 'language']].reset_index(drop=True)

    # print(df1)
    # print('===================')
    # print(df2)
    return df1, df2


if __name__ == "__main__":
    main()
