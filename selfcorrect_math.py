###coding:utf-8
###数学问题的self correction
####****仅限制在固定的one shot和zero shot learning条件下，进行7b以下llm的inference***
####few shot prom from paper: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
fewshot_8 = '''These are some examples for solving math problem:
    Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done,
    there will be 21 trees. How many trees did the grove workers plant today?
    A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted.
    So, they must have planted 21 - 15 = 6 trees. The answer is 6.
    Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
    A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.
    Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
    A: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74
    chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.
    Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops
    did Jason give to Denny?
    A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of
    lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.
    Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he
    have now?
    A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so
    in total he has 7 + 2 = 9 toys. The answer is 9.
    Q: There were nine computers in the server room. Five more computers were installed each day, from
    monday to thursday. How many computers are now in the server room?
    A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 =
    20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers.
    The answer is 29.
    Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many
    golf balls did he have at the end of wednesday?
    A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On
    Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.
    Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
    A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in
    beginning, so now she has $23 - $15 = $8. The answer is 8.\n'''


hard_fewshot = '''These are some examples for solving math problem:
Q:Frank was reading through his favorite book. The book had 2 chapters each with 405 pages. It took frank 664 days to 
finish the book. How many days did it token for reading one chapter?
A: the answer is 664.0 / 2.0, so it is 332.0
Q:The Razorback t-shirt shop sells each t-shirt for $ 51 dollars. During the Arkansas and Texas tech game they offered
a discount of $ 8 per t-shirt and sold 130 t-shirts. How much money did they make from selling the t-shirts?
A:t-shirt price with discount is 51.0 - 8.0, they sell 130 t-shirts, so totally they make ( 51.0 - 8.0 ) * 130.0 dollars, therefor the answer is $5590.0.
Q:Paige raised 7 goldfish and 12 catfish in the pond but stray cats loved eating them. Now she has 15 left.How many fishes disappeared?
A:Paige raised 7.0 + 12.0 fishes in the pond, now she has 15 left, so the cats eat ( 7.0 + 12.0 ) - 15.0 fishes, therefor the answer is 4.0 fishes.
Q:Faye had 35 packs of pencils each one having 4 pencils. She was placing her pencils into rows with 2 pencils in each row.
How many rows could she make?
A:Faye had 35.0 * 4.0 pencils in all, 2 pencils in each row, so there are 35.0 * 4.0 / 2.0 rows, therefore the answer is 70.0.
Q:The Razorback shop makes $ 192 dollars off each t-shirt and $ 34 off each jersey. During the Arkansas and Texas tech game they sold 157 t-shirts and 19 jerseys.
How much more does a t-shirt cost than a jersey?
A:the shop makes $192.0 each t-shirt, so the t-shirt price is $192.0, then we can see the jersey price is 34.0, 
therefor t-shirt cost $192.0 - $34.0 more, the answer is 158.0.
Q:6 birds and 3 storks were sitting on the fence. 2 more storks came to join them.How many more birds than storks are sitting on the fence?
A:6 birds and 3 storks were sitting on the fence,2 more storks came , then there were 3.0+2.0 storks, 
so birds are 6.0 - ( 3.0 + 2.0 ) more than storks. the answer is 1.0
Q:Paul had 2 books. After selling some in a garage sale he bought 150 new ones. If he has 58 books now
How many books did he sell?
A:Paul had 2 books, then he bought 150 new ones, so he should have ( 2.0 + 150.0 ) books, but he sell some, 
then he had 58.0 books, so he selled ( 2.0 + 150.0 ) - 58.0 books. the answer is 94.0.
Q:Each Ferris wheel in paradise park has 19 seats. Each seat in a Ferris wheel can hold 15 people.How many people can ride 20 Ferris wheels at the same time?
A:One Ferris wheel has 19 seats, one seat hold 15 people, so one Ferris wheel can hold 19.0 * 15.0 people. there are 20 Ferris wheels 
then ( 19.0 * 15.0 ) * 20.0 people can ride on the same time. the answer is 5700.
'''

contrastive_prompt = '''The following are examples of math problems and their solutions, which have wrong and right answers.
 Please refer to the right answer to solve the new problem, and also avoid mistakes in the wrong answers.
 
Question1: Leah had 32 chocolates and her sister had 42.If they ate 35, how many pieces do they have left in total?

Right Answer: Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74.
After eating 35, they had 74 - 35 = 39 pieces left in total.

Wrong Answer: Originally, Leah had 32 + 42 = 74 chocolates and her sister had 32. So in total they had 74 -
35 = 39. After eating 35, they had 42 pieces left in total.

Explanation: The reason this answer is incorrect is because it states that Leah's sister had 32 chocolates, 
which is wrong. The question clearly mentions that Leah's sister had 42 chocolates. 
|EOS|

Question2:Frank was reading through his favorite book. The book had 2 chapters each with 405 pages. It took frank 664 days to 
finish the book. How many days did it token for reading one chapter?

Right Answer: The book had 2 chapters, and each chapter had 405 pages. It took Frank 664 days to finish the entire book.
 Since there were 2 chapters, we can divide the total number of days (664) by the number of chapters (2) to find the number
  of days it took to read one chapter. Therefore, it took Frank 664 days / 2 chapters = 332 days to read one chapter.

Wrong Answer: The book had 2 chapters, and each chapter had 405 pages. So the total number of pages in the book was 810 pages.
 It took Frank 664 days to finish the entire book. Since the book had 810 pages, and Frank took 664 days to finish it, 
 he must have read 810 pages in 664 days. Therefore, it took Frank 664 days to read one chapter.

Explanation: In the wrong answer, the critical mistake is that Frank read the entire book of 810 pages in 664 days
. However, this contradicts the given information that the book had 2 chapters, and Frank finished the entire book in 664 days.
|EOS|

Question3:The Razorback t-shirt shop sells each t-shirt for $ 51 dollars. During the Arkansas and Texas tech game they offered
a discount of $ 8 per t-shirt and sold 130 t-shirts. How much money did they make from selling the t-shirts?

Right Answer:t-shirt price with discount is 51.0 - 8.0, they sell 130 t-shirts, so totally they make ( 51.0 - 8.0 ) * 130.0 dollars, 
therefor the answer is $5590.0.

Wrong Answer: The Razorback t-shirt shop sells each t-shirt for $51. During the Arkansas and Texas Tech game, they 
offered a discount of $8 per t-shirt and sold 130 t-shirts. Therefore, the total money they made is $51 × 130 = $6,630, 
as they sold each t-shirt for the original price of $51.

Explanation: In the wrong answer, the total money they made is $51 x 130 = $6,630. 
This is incorrect because it does not consider the $8 discount offered during the game.
|EOS|

Question4:Paige raised 7 goldfish and 12 catfish in the pond but stray cats loved eating them. Now she has 15 left.How many fishes disappeared?

Right Answer:Paige raised 7.0 + 12.0 fishes in the pond, now she has 15 left, so the cats eat ( 7.0 + 12.0 ) - 15.0 fishes, 
therefor the answer is 4.0 fishes.

Wrong Answer: Initially, Paige had a total of 7 goldfish + 12 catfish = 19 fish.
Now, she has 15 fish left, which means 19 - 15 = 4 fish are left.Therefore, 19 fish disappeared from Paige's pond.

Explanation: The wrong answer incorrectly assumes that the remaining 15 fish are the ones that disappeared, 
instead of the ones that are still left in the pond. The correct approach is to subtract the remaining fish 15 
from the initial number of fish 19 to find the number of fish that disappeared 4.
|EOS|
'''

prompt_d ={}
####plan and solve
prompt_d['sim_cot'] = "{}.Let's think step by step."

prompt_d['sim_cot_sc'] = "{}.Let's first understand the problem and devise a plan to solve the problem. " \
             "Then, let's carry out the plan to solve the problem step by step."

prompt_d['sim_cot_ps'] = "{}.Let's first understand the problem, extract relevant variables and their corresponding numerals, " \
             "and make and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables " \
             "(pay attention to correct numerical calculation and commonsense), " \
             "solve the problem step by step, and show the answer."

###self correction
prompt_d['selfcorrect'] = "{}.Let's first understand the problem and devise a plan to solve the problem. " \
             "Then, let's carry out the plan to solve the problem step by step."

###faithfual cot
prompt_d['faith_cot'] = "{}.Let's first understand the problem, extract relevant variables and their corresponding numerals, " \
             "and make and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables " \
             "(pay attention to correct numerical calculation and commonsense), " \
             "solve the problem step by step, and show the answer."

###self prompter
prompt_d['self_prompt'] = '''
Question:
Adam had 5 apples. He ate 2 of them for breakfast.How many apples will he have left if he eats 1 more?

Answer:
1.transfer math problem to known condition and question.
Text known condition: "Adam had 5 apples. He ate 2 of them for breakfast."
Text question: "How many apples will he have left if he eats 1 more?"

2.transfer Text format to Symbolic format,then map the numbers and symbolics using a key-value mapping
Symbolic known condition: "Adam had w apples. He ate x of them for breakfast."
Symbolic question: "How many apples will he have left if he eats y more?"
Mapping: {{w:5,x:2,y:1}}

2. using "Let's think step by step" to solve the Text question, then get the Text answer as a number.
Text inferring: "After eating 2 apples,Adam will have 3 left. if he eats 1 more, there will be 2 apple left."
Text answer: 2

4.transfer the Text inferring to Symbolic format, with refer to the Mapping, get the Symbolic answer;
Symbolic inferring: "Adam had w apples,After eating x apples,Adam will have w-x left. if he eats y more, there will be w-x-y apple left."
Symbolic Answer: 2

5. collect the formulas in Symbolic inferring.
Symbolic formular: [w-x, w-x-y]


Please follow the above example to solve the math problem:
Question:
{}

Answer:
'''

prompt_d['reverse_cot'] = '''### Instruction:
Given a math problem, please solve it in 6 steps: 
1. list the known conditions. transfer the values in the known conditions to numeric.
2. explain the objective of the problem.
3. recursively list the intermediate problems should be known to solve the problem.
4. solve the intermediate problems steps by steps if necessary. Be careful when copy the numbers from Step 1.
5. check the correctness of the intermediate solutions, especially pay attention to the dependency between the variables. 
for example: increase/decrease from what, ratio of what, more/less of what, etc.
6. sum up the intermediate results, then solve the problem.

Below is an example:
Q:Xiao Ming bought some apples. The price of apples is two yuan per kilo. He paid ten yuan and received 2 yuan in change. How many kilos of apples did Xiao Ming buy?
A:1: find Known conditions,transfer the values to numeric:
The price of apples is 2 yuan per kilo.
Xiao Ming paid 10 yuan.
Xiao Ming received 2 yuan in change.
2:explain the problem: How many kilos of apples did Xiao Ming buy? 
we know the price of the apples, we can find money Xiao Ming paid, then we can solve the problem.
3: recursively list the intermediate problems should be known to solve the problem.
how much money xiaoming paid for buying apples?
4: solve the intermediate problems in the logic order.
xiaoming paid 10-2=8 yuan for buying apples.
5: calculation 10-2=8 is correct.
6: sum up to solve the problem
So Xiao Ming bought 8/2=4 kilos of apples.

Q:{}

A:
'''

prompt_d['reverse_symbolic'] = '''### Instruction:
Given a math problem, please solve it in 6 steps: 
1. list the known conditions. transfer the values in the known conditions to numeric. replacing the numeric entries with variables
2. explain the objective of the problem.
3. recursively list the intermediate problems should be known to solve the problem.
4. solve the intermediate problems steps by steps if necessary. Be careful when copy the numbers from Step 1.
5. check the correctness of the intermediate solutions, especially pay attention to the dependency between the variables. 
for example: increase/decrease from what, ratio of what, more/less of what, etc.
6. sum up the intermediate results, then solve the problem.

Below is an example:
Q:Xiao Ming bought some apples. The price of apples is two yuan per kilo. He paid ten yuan and received 2 yuan in change. How many kilos of apples did Xiao Ming buy?
A:
1: find Known conditions,transfer the values to numeric:
The price of apples is 2 yuan per kilo.
Xiao Ming paid 10 yuan.
Xiao Ming received 2 yuan in change.
2:explain the problem: How many kilos of apples did Xiao Ming buy? 
we know the price of the apples, we can find money Xiao Ming paid, then we can solve the problem.
3: recursively list the intermediate problems should be known to solve the problem.
how much money xiaoming paid for buying apples?
4: solve the intermediate problems in the logic order.
xiaoming paid 10-2=8 yuan for buying apples.
5: calculation 10-2=8 is correct.
6: sum up to solve the problem
So Xiao Ming bought 8/2=4 kilos of apples.

Q:{}

A:

'''

prompt_d['mathprompt'] = '''
Given a math problem, please solve it step by step: 
1. transforming the question into its Algebraic form by replacing the numeric entries with variables using a key-value mapping
2. Write a mathematical equation and generate the answer format starting with ‘Answer =’
3. Write a Python function
4. verify consensus of algebraic answer and pythonic answer
5. return answer

here is an example:
Q: At a restaurant, each adult meal costs $5 and kids eat free. If a group of 15 people came in and 8 were kids, how much would it cost for the group to eat?
Qt: at a restaurant, each adult meal costs A and kids eat free. if a group of B people came in and C were kids, how much would it cost for the group to eat?
Mapping: {A:5, B:15, C:8}
# Algebraic expression output
Answer = A*(B-C)
# Python expression output
def total_price(A, B, C):
    return A * (B-C)
Algebraic-answer = 35
Pythonic-answer = 35
Answer = 35

Q:{}
A:
'''

####1示例，4步解答
prompt_cot = """
### Instruction:
Given a math problem, please solve it in 4 steps: 
1. Understand the problem and identify the key information.
2. Build a mathematical model based on the key information.
3. using mathematical methods to solve the model
4. interpret the results into natural language

Below is an example for solving a math problem:
Q:Xiao Ming bought some apples. The price of apples is 2 yuan per kilo. He paid 10 yuan and received 2 yuan in change. How many kilos of apples did Xiao Ming buy?
A: 
1: Understand the problem and identify the key information
Known conditions:
The price of apples is 2 yuan per kilo.
Xiao Ming paid 10 yuan.
Xiao Ming received 2 yuan in change.
Solution objective: How many kilos of apples did Xiao Ming buy?
2: Build a model. Build a mathematical model based on the information in the problem
Let Xiaoming buy x kilos of apples.
The amount Xiao Ming should pay is 2 * x yuan.
Since Xiao Ming received 2 yuan in change, the actual amount he paid was 10-2 = 8 yuan.
Therefore, we can establish the equation: 2 * x = 8
3: Solve the model using mathematical methods to solve the model
Divide both sides of the equation by 2 to get x = 4
4: Answer the question and interpret the results into natural language
So Xiao Ming bought 4 kilos of apples.

Q:{}

A:

"""
import os, re, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import argparse
from soft_embedding import Softcot
from transformers import get_linear_schedule_with_warmup
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=str, default='data\gsm8k_main_test.json')
parser.add_argument('--outfile', type=str, default='none', required=True)
parser.add_argument('--model_path', type=str, default='E:\pretrained\mistral7b')
parser.add_argument('--shot_RAG', type=int, default=0, help='RAG few shot num ')
parser.add_argument('--shot_num', type=int, default=0, help='few shot num')
parser.add_argument('--promptex', type=str, default=prompt_cot, help='choose prompt from sim cot prompt_cot etc', required=True)
args = parser.parse_args()
# model_path = 'E:\pretrained/tinyllama1b'

####基于语义相似性的检索
def get_rag(text, zeroshotmodel):
    with open('logic_samples.json', 'r', encoding='utf8') as fi:####read tsv or txt
        lines = fi.readlines()
    preds = []

    for t in lines:
        prob = get_semantic_entail(zeroshotmodel, t, text)
        preds.append(prob)
    simidx = np.argsort(preds)[-5:]
    return '\n'.join([lines[idx] for idx in simidx])+'\n'

###测试gsm8k,本数据集有推理过程，假定推理过程是正确的，并且可以用大模型获得(即使不正确，相似的题目应该有相似的推理)
def test_gsm8k():
    with open('gsm8k_main_train.json', 'r', encoding='utf8') as fi:####read tsv or txt
        lines = fi.readlines()

def pro_logic(llm, question):
    prompt = '''1. list the known conditions. transfer the values in the known conditions to numeric.
                2. explain the objective of the problem.
                3. recursively list the intermediate problems should be known to solve the problem.
                4. transfer the intermediate problems to algebraic format'''

###逻辑相似性检索
def logic_rag():
    print()

####测试100个例子, args包括infile outfile modelpath prompt等参数
def test_classify100(args, nums=100):
    if not os.path.exists(args.infile):
        exit('NO input file found')

    with open(args.infile, 'r', encoding='utf8') as fi:####read tsv or txt
        lines = fi.readlines()
    # else:
    #     with open('data/svamp_train.json', 'r', encoding='utf-8') as fi:
    #         lines = json.load(fi)
    print('check lines ', len(lines))
    if nums == 0:
        nums = int(len(lines))

    ####output路径
    output_data_path = args.outfile###test_data_path.replace('.tsv', '.json')
    generated_responses = []
    if os.path.exists(output_data_path):
        with open(output_data_path, 'r', encoding='utf8') as fin:
            for line in fin:
                generated_responses.append(json.loads(line))
        print(f"Loaded {len(generated_responses)} generated responses")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
    model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_4bit=True,  ##torch_dtype=torch.int8,
                                                 trust_remote_code=True)
    model = model.eval()

    ###check llm generate config###max_new_tokens do sample可以提高样本的多样性，设置eos可以控制提前结束
    gen_kwargs = {"max_new_tokens": 384, "top_p": 0.95, "temperature": 0.1, "top_k": 30,'early_stopping': True,'eos_token_id': tokenizer.eos_token_id,
                  "do_sample": True, 'repetition_penalty': 1.15,'pad_token_id': tokenizer.eos_token_id}####
    # gen_kwargs = {"max_new_tokens": 384, 'num_beams':3, 'early_stopping': True,'eos_token_id': tokenizer.eos_token_id,
    #               "do_sample": False, 'repetition_penalty': 1.15,'pad_token_id': tokenizer.eos_token_id, 'no_repeat_ngram_size':5}####
    gen_kwargs['eos_token_id'] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    ###, "stop_sequences": '<|END|>', "num_return_sequences": 5num控制返回数量，不如单次返回一个答案

    # unwind broken decapoda-research config，貌似有问题，大模型tokenizer中已经定义好了eos pad bos
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    with open(output_data_path, 'a', encoding='utf8') as fout:
        for ids, data in enumerate(lines):
            data = json.loads(data)
            if ids >= nums:###总测试数量限制
                exit()
            if ids < len(generated_responses):####跳过
                continue
            print('processing the', ids, 'th sample...\n')

            if 'question' in data.keys():####gsm8k为字符串
                ent = data['question']
                ans = data['answer']
            else:
                ent = data['Body']+data['Question']
                ans = data['Answer']

            prompt = prompt_d[args.promptex].format(ent)
            # print('check prompt', prompt)
            ###few shot
            if args.shot_num == 1:
                prompt = fewshot_8 + prompt
            if args.shot_num == 2:
                prompt = hard_fewshot + prompt
            if args.shot_num == 3:
                prompt = contrastive_prompt + prompt
            ###RAG few shot
            if args.shot_RAG > 0:
                prompt = get_rag(ent, model)+prompt

            inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
            if not 'sim' in args.promptex:
                gen_kwargs['max_new_tokens'] = int(512)###生成长度
            # print('check length ', gen_kwargs['max_length'])
            res = []
            for _ in range(5):####给出5个结果,self correction 耗时太长，不比较
                generate_ids = model.generate(**inputs, **gen_kwargs)
                if not any(end_token in generate_ids[0] for end_token in gen_kwargs['eos_token_id']):
                    print('entering re generate data ###############')
                    gen_kwargs['max_new_tokens'] = gen_kwargs['max_new_tokens'] + 128  ###生成长度
                    generate_ids = model.generate(**inputs, **gen_kwargs)
                for gen_out in generate_ids:
                    # response = tokenizer.batch_decode(gen_out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    response = tokenizer.decode(gen_out, skip_special_tokens=True)
                #     response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                # response = response.split(end_token)[0]

                    response = response.replace(prompt, "").strip()####消去已知条件
                    print('responese ', response)
                    res.append(response)
            generated_responses.append({"answer": ans,
                           "predict": res})
            data['predict'] = res
            fout.write(json.dumps(data) + "\n")  ###加入indent就不行


import numpy as np
def cosine_similarity(vector1, vector2):
  """计算两个向量的余弦相似度。

  Args:
    vector1: 一维数组。
    vector2: 一维数组。

  Returns:
    余弦相似度。
  """

  dot_product = np.dot(vector1, vector2)
  norm1 = np.linalg.norm(vector1)
  norm2 = np.linalg.norm(vector2)
  return dot_product / (norm1 * norm2)
###通过语义蕴含进行分类
def get_semantic_entail(sentmodel, sentence1, sentence2):

    probability = cosine_similarity(sentmodel.encode(sentence1), sentmodel.encode(sentence2))
    return probability
    # print("Entailment:", probability)
###通过text和label的语义蕴含，得到分类结果
def zeroshot_classify(zeroshotmodel, texts, label):
    preds = []

    for t in texts:
        prob = get_semantic_entail(zeroshotmodel, t, label)
        preds.append([1-prob, prob])

    return preds

###按当前步骤，截取部分答案作为label,一共
def cut_ans(ans, epoch, cotsteps):
    if epoch % cotsteps == cotsteps-1:###最后一步，step以1开头
        return ans
    else:
        affix = 'Step ' + str(epoch % cotsteps + 2)###第0步，取Step2之前的部分
        return ans.split(affix)[0]

def train_softcot(model_path):
    sentmodel = SentenceTransformer("E:\pretrained\paramulti_mpnetv2")
    grade_acc = 8
    cotsteps = 4

    with open(args.infile, 'r', encoding='utf8') as fi:####read tsv or txt
        lines = fi.readlines()
    print('check lines ', len(lines))

    freeze_model = AutoModelForCausalLM.from_pretrained(model_path,  ##load_in_4bit=True,  ##torch_dtype=torch.int8,
                                                        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Softcot(freeze_model.get_input_embeddings(),
                          n_tokens=20,
                          layers=5,
                          initialize_from_vocab=True)

    loss_fct = torch.nn.BCEWithLogitsLoss()

    ###check llm generate config
    gen_kwargs = {"max_length": 512 + 120, "top_p": 0.9, "temperature": 0.7, "do_sample": True,
                  "repetition_penalty": 1.0}
    # "trainable params: 300,288 || all params: 559,514,880 || trainable%: 0.05366935013417338"

    lr = 3e-2
    num_epochs = 20 * cotsteps
    device = "cuda"
    model = model.to(device)

    for epoch in range(num_epochs):
        lay_id = epoch % cotsteps  ###cot层数
        ####更改学习率，实现分层训练
        param_list = []
        for i in range(cotsteps):
            if i == lay_id:
                param_list.append({'params': model.layers[i].parameters(), 'lr': lr})
            else:
                param_list.append({'params': model.layers[i].parameters(), 'lr': 0})
        optimizer = torch.optim.AdamW(param_list)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0,
                                                       num_training_steps=int(len(lines) * 0.1), )

        model.train()
        total_loss = 0
        for step, data in enumerate(lines):
            # batch = {k: v.to(device) for k, v in batch.items()}
            ldata = re.split('\t', data)
            ent = ldata[1]
            ans = ldata[0]
            ans = cut_ans(ans, epoch, cotsteps)
            prompt = prompt_cot.format(ent)

            inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
            gen_kwargs['max_length'] = int(len(inputs["input_ids"][0]) + 384)  ###生成长度
            generate_ids = model.generate(**inputs, **gen_kwargs)
            response = \
            tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            response = response.replace(prompt, "").strip()
            print(response + '\n')

            preds = torch.Tensor(
                zeroshot_classify(sentmodel, response, ans)).requires_grad_()  ###文本分类，判断resposes是否对应true
            print('check preds', preds)
            # loss = outputs.loss
            loss = loss_fct(preds.to(device), 1)####一致则为1
            total_loss += loss.detach().float()

            if step % grade_acc == 0:###累积误差
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

if __name__ == '__main__':
    # model_path = 'E:\pretrained\mistral7b'

    test_classify100(args)
    # train_softcot(model_path)