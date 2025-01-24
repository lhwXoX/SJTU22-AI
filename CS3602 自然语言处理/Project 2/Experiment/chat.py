import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import BertModel, BertTokenizer
from peft import PeftModel
from sentence_transformers import util
import os

class KnowledgeBase:
    def __init__(self, bert_path):
        # 加载 BERT 模型和 Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.model = BertModel.from_pretrained(bert_path)
        
    def process_knowledge(self, knowledge_file):
        # 加载知识库，每行作为一个知识点
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            knowledge_points = [line.strip() for line in f if line.strip()]
        # 为每行生成嵌入
        embeddings = []
        for knowledge in knowledge_points:
            embeddings.append(self._generate_embedding(knowledge))
        # 返回知识点和其对应的嵌入
        return knowledge_points, torch.stack(embeddings)

    def _generate_embedding(self, text):
        # Tokenize 文本
        inputs = self.tokenizer(text, return_tensors="pt")
        # 通过模型生成嵌入
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 使用 [CLS] token 的嵌入作为句子嵌入
        sentence_embedding = outputs.last_hidden_state[:, 0, :]
        return sentence_embedding.squeeze(0)

def retrieve_knowledge(question, knowledge_points, embeddings, kb, top_k=1):
    question_embedding = kb._generate_embedding(question)
    scores = util.cos_sim(question_embedding, embeddings)[0]
    #best_index = torch.argmax(scores).item()
    top_k_scores, top_k_indices = torch.topk(scores, top_k)
    top_k_results = [(knowledge_points[idx], scores.item()) for idx, scores in zip(top_k_indices, top_k_scores) ]
    return top_k_results
    
class Chatbot:
    def __init__(self, base_model_path, model_name, bert_path, knowledge_file, max_history_length=1024):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
        self.model = PeftModel.from_pretrained(self.model, model_name).to("cuda:0")
        self.max_history_length = max_history_length
        self.history = []
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.kb = KnowledgeBase(bert_path)
        self.knowledge_points, self.embeddings = self.kb.process_knowledge(knowledge_file)
        
    def process_input(self, user_name, user_input, flag: bool, flag_idt: bool, tone_text: str, top_k: int = 1, identity: str = 'a knowledgeable assistant'):
        if user_input == "\\quit" or user_name == "\\quit":
            print("会话结束。")
            return False  # 结束程序
        elif user_input == "\\newsession" or user_name == "\\newsession":
            self.history = []  # 清空对话历史
            print("开启新的对话。")
        else:
            self.generate_response(user_name, user_input, flag, flag_idt, tone_text, top_k, identity)
        return True

    def generate_response(self, user_name, user_input, flag: bool, flag_idt: bool, tone_text: str, top_k: int = 1, identity: str = 'a knowledgeable assistant'):
        # 构造对话历史
        top_k_results = retrieve_knowledge(user_input, self.knowledge_points, self.embeddings, self.kb, top_k)
        if identity == 'a knowledgeable assistant':
            identity_text = '.'
        else:
            identity_text = ' in a way that suits your identity.'
        relevent_knowledge = []
        if flag:
            for idx, (knowledge, score) in enumerate(top_k_results, 1):
                relevent_knowledge.append(knowledge)
                print(f'检索到第{idx}条/共{top_k}条: {knowledge}\n分数: {score}')
            relevent_knowledge = ' '.join(relevent_knowledge)
        if flag_idt:
            formatted_user_input = f"Questioner's name: {user_name}. Current question: {user_input}. "
        else:
            formatted_user_input = f"Current question: {user_input}. "
        #formatted_user_input = f"Current talk: {user_input}."
        if len(self.history) > 0:
            for i in range(len(self.history)):
                self.history[i] = self.history[i].replace('Current question', 'History talk')
        self.history.append(formatted_user_input)

        # 确保对话历史不超出模型的最大长度
        while len(self.tokenizer("".join(self.history), return_tensors="pt")["input_ids"][0]) > self.max_history_length:
            self.history.pop(0)  # 移除最早的对话
        print("".join(self.history))
        # 构建输入序列
        #print("历史: ", "".join(self.history))
        if len(self.history) == 1:
            print("first")
            input_text = (
            f"<|im_start|>system\nYou are {identity}. Please give a detailed response to the following question{identity_text}<|im_end|>\n<|im_start|>user\nAnswer to current question based on questioner's name{tone_text} "
            + ''.join(self.history)
            + "<|im_end|>\n<|im_start|>assistant\n"
            ) if flag_idt else (
            f"<|im_start|>system\nYou are {identity}. Please give a detailed response to the following question{identity_text}<|im_end|>\n<|im_start|>user\nAnswer to current question{tone_text} "
            + ''.join(self.history)
            + "<|im_end|>\n<|im_start|>assistant\n"
            )
            input_text_with_injection = (
            f"<|im_start|>system\nYou are {identity}. Please give a detailed response to the following question based on external documents{identity_text}<|im_end|>\n<|im_start|>user\nExternal documents: <|object_ref_start|>{relevent_knowledge}<|object_ref_end|> Answer to current question based on questioner's name and external documents{tone_text} "
            +''.join(self.history)
            + "<|im_end|>\n<|im_start|>assistant\n"
            ) if flag_idt else (
            f"<|im_start|>system\nYou are {identity}. Please give a detailed response to the following question based on external documents{identity_text}<|im_end|>\n<|im_start|>user\nExternal documents: <|object_ref_start|>{relevent_knowledge}<|object_ref_end|> Answer to current question based on external documents{tone_text} "
            +''.join(self.history)
            + "<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            input_text = (
            f"<|im_start|>system\nYou are {identity}. Please give a detailed response to the following question{identity_text}<|im_end|>\n<|im_start|>user\nAnswer to current question based on questioner's name and histroy talk{tone_text} "
            + '<|object_ref_start|>'+"".join(self.history)+'<|object_ref_end|>'
            + "<|im_end|>\n<|im_start|>assistant\n"
            ) if flag_idt else (
            f"<|im_start|>system\nYou are {identity}. Please give a detailed response to the following question{identity_text}<|im_end|>\n<|im_start|>user\nAnswer to current question based on histroy talk{tone_text} "
            + '<|object_ref_start|>'+ "".join(self.history)+'<|object_ref_end|>'
            + "<|im_end|>\n<|im_start|>assistant\n"
            )
            input_text_with_injection = (
            f"<|im_start|>system\nYou are {identity}. Please give a detailed response to the following question based on external documents{identity_text}<|im_end|>\n<|im_start|>user\nExternal documents: <|object_ref_start|>{relevent_knowledge}<|object_ref_end|> Answer to current question based on questioner's name, histroy talk and external documents{tone_text} "
            + '<|object_ref_start|>'+"".join(self.history)+'<|object_ref_end|>'
            + "<|im_end|>\n<|im_start|>assistant\n"
            ) if flag_idt else (
            f"<|im_start|>system\nYou are {identity}. Please give a detailed response to the following question based on external documents{identity_text}<|im_end|>\n<|im_start|>user\nExternal documents: <|object_ref_start|>{relevent_knowledge}<|object_ref_end|> Answer to current question based on histroy talk and external documents{tone_text} "
            + '<|object_ref_start|>'+"".join(self.history)+'<|object_ref_end|>'
            + "<|im_end|>\n<|im_start|>assistant\n"
            )
        print("输入: ", input_text_with_injection if flag else input_text)
        
        # 模型推理
        if flag:
            inputs = self.tokenizer(input_text_with_injection, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=1000,  # 限制回复的长度
            streamer = self.streamer, # 流式输出
            )
        # 解码回复
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("assistant\n")[-1].strip()
        response = response.split("<|im_end|>")[0].strip()
        # 记录回复
        self.history.append(f" Assistant's history response: {response}\n")

    def chat(self):
        print("聊天机器人已启动。输入 \\quit 结束会话，输入 \\newsession 开启新会话。")
        top_k = 1
        choice = input("是否使用外部知识增强？(Yes/No)\n")
        if choice == 'Yes':
            flag_ext = True
            top_k = int(input("检索几条外部知识？\n"))
        elif choice == 'No':
            flag_ext = False
        else:
            raise Exception('Yes or No only.')
        choice_identity = input('是否给予虚拟人身份？(Yes/No)\n')
        if choice_identity == 'No':
            role = 'a knowledgeable assistant'
        elif choice_identity == 'Yes':
            role = input('请输入虚拟人身份(例如: a king in the kingdom):\n')
        else:
            raise Exception('Yes or No only.')
        identity = input("是否输入说话人？(Yes/No)\n")
        tone = input("机器人以什么口吻聊天(例: humorous, 不需要请输入No)\n")
        if tone == 'No':
            tone_text = '.'
        else:
            tone_text = f' in a {tone} tone.'
        if identity == 'Yes':
            flag_idt = True
        elif identity == 'No':
            flag_idt = False
        else:
            raise Exception("Yes or No only.")
    
        print(f'外部知识: {flag_ext}\n说话人: {flag_idt}\n口吻: {tone}')
        if flag_idt:
            while True:
                user_name = input("Who are you? (input user name):\n")
                if user_name == "\\quit":
                    print("会话结束。")
                    return False
                if user_name == "\\newsession":
                    self.process_input(user_name, None, flag_ext, flag_idt, tone_text, top_k, role)
                    continue
                user_input = input("Feel free to ask me! (input question):\n")
                if not self.process_input(user_name, user_input, flag_ext, flag_idt, tone_text, top_k, role):
                    break
        else:
            while True:
                user_name = 'Alice'
                user_input = input("Feel free to ask me! (input question):\n")
                if not self.process_input(user_name, user_input, flag_ext, flag_idt, tone_text, top_k, role):
                    break

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_model_path = os.path.join(BASE_DIR, 'Qwen2.5-1.5B')
model_name = os.path.join(BASE_DIR, 'output-r=16', 'checkpoint-51760') # 只支持微调的模型
bert_path = os.path.join(BASE_DIR, 'bert-base-cased')
Knowledge_file = os.path.join(BASE_DIR, 'knowledge_base.txt')

chatbot = Chatbot(base_model_path, model_name, bert_path, Knowledge_file)
chatbot.chat()
