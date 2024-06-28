###coding:utf-8
###数学问题的self correction RAG enhancement
####****仅限制在固定的one shot和zero shot learning条件下，进行7b以下llm的inference***
###测试gsm8k,本数据集有推理过程，假定推理过程是正确的，并且可以用大模型获得(即使不正确，相似的题目应该有相似的推理)
prompt_d ={}

prompt_d['sim_cot_sc'] = "{}.Let's first understand the problem and devise a plan to solve the problem. " \
             "Then, let's carry out the plan to solve the problem step by step."

prompt_d['sim_cot_ps'] = "{}.Let's first understand the problem, extract relevant variables and their corresponding numerals, " \
             "and make and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables " \
             "(pay attention to correct numerical calculation and commonsense), " \
             "solve the problem step by step, and show the answer."



import os, re, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import argparse
from soft_embedding import Softcot
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=str, default='data\gsm8k_test_formu.json')
parser.add_argument('--outfile', type=str, default='data\gsm8k_test_shotrag.json')
parser.add_argument('--model_path', type=str, default='E:\pretrained\mistral7b')
parser.add_argument('--RAG_path', type=str, default='data\gsm8k_train_formu.json', help='RAG few shot file path')
parser.add_argument('--shot_num', type=int, default=8, help='few shot num')
parser.add_argument('--promptex', type=str, default='sim_cot_sc', help='choose prompt from sim cot prompt_cot etc')
args = parser.parse_args()
# model_path = 'E:\pretrained/tinyllama1b'
from sentence_transformers import SentenceTransformer, util
from cal_accuracy import extract_answer
if os.path.exists("E:\pretrained\para_minilmL12v2"):
    sent_model = SentenceTransformer("E:\pretrained\para_minilmL12v2")
else:
    sent_model = SentenceTransformer("../pretrained/para_minilmL12v2")
###使用问题的相似性表征
def ques_similarity(s1,s2, sentmodel):
    # Compute embedding for both lists
    embeddings1 = sentmodel.encode([s1, s2], convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings1)
    return cosine_scores[0][1].cpu()

import difflib
def string_similarity(s1, s2):
    """
    计算两个字符串的归一化编辑距离相似性
    Returns:
        float: 两个字符串的相似性分数 (0 - 1)
    """
    # 计算编辑距离
    distance = difflib.SequenceMatcher(None, s1, s2).ratio()

    return distance

###输入[formu, formu], formu为str,假定样本足够大，则必然有相关逻辑的计算
def get_log_sim(forls1,forls2):
    if len(forls1) == 1 and len(forls2) == 1:
        return string_similarity(forls1[0], forls2[0])
    return string_similarity(';'.join(forls1[1:-1]), ':'.join(forls2[1:-1]))+string_similarity(forls1[-1],forls2[-1])

###表达式规范化，数值或字符转# 替换匹配到的数字为单个"@"字符
def alg_nor(input_str):
    ###分别替换公式中的单词或数字
    pattern = r'\d+'####r'-?\d+\.?\d+'  ##数字只是整数，####保留小数，负数，百分数等，例如-100，-2.3，1.11等方便对比
    result = re.sub(pattern, '@', input_str)
    pattern = r'[a-zA-Z]+'##
    result = re.sub(pattern, '@', result)

    # ###不规范小数点，不保留
    result = re.sub(r'@\.@', r'@', result)###
    ###不规范推理
    result = re.sub(r'=.*=', r'=', result)
    ###不规范乘法符号，x(x)
    result = re.sub(r'@\(', r'@*(', result)
    result = re.sub(r'@@', r'@*@', result)
    return result

####提取运算式
def extra_alg(text):
    ####加减乘除运算符
    # patterns = [r'[0-9a-zA-Z(]+\s[+\-*/=].*?[+\-*/=]\s[$0-9a-zA-Z]+', r'[0-9a-zA-Z(]+[+\-*/=].*?[+\-*/=][$0-9a-zA-Z]+']###以字母数字+空格+运算符开头，以运算符+空格+字母数字结尾，.*表示忽略中间
    # pattern = r'(?:\d+|\w|\()((?!([\w\s]+))[\w\d\*\+\-\/\(\)=].*)(?:\d+|\w|\))'###(?!([\w\s]+))否定的前瞻断言,确保后面不会出现单词空格单词的形式
    pattern = r'(?:\d+|\w|\()[\w\d\*\+\-\/\(\)\=]+?:\d+|\w|\)'##(?<!([\w\s|\n]))
    pattern = r'([\d+\w+\(][*+-/=][\d+\w+\)]).*?'
    # patterns = [r"[0-9X/\-\+\*\$]+ = [0-9X/\-\+\*\$]+"]
    # 使用re.findall()函数查找所有匹配的运算表达式
    # expressions = []
    # for pattern in patterns:
    results = re.findall(pattern, text)
    print('check result ', results)
    expressions = [lin for lin in results if len(lin) > 3]
    return expressions

###英文提取公式，注意有连词例如'regular-priced'，
def extra_alg2(text):
    text = re.sub(r'\s*([\+\-\*/=])\s*', r'\1', text)####消除公式中的空格
    # print('check without space', text)
    # 使用空格分割字符串
    words = text.split()
    # print('check words', words)
    algebraic_expression = []
    for word in words:
        if any(op in word for op in ['+', '-', '*', '/', '=']):
            if len(re.findall(r'[0-9]', word)) > 0 and '=' in word:####简化，只抽取含有等号的公式
                algebraic_expression.append(word)
        # elif algebraic_expression and algebraic_expression[-1].isdigit() and word[0].isdigit():
        #     algebraic_expression += "*" + word
        # elif algebraic_expression and algebraic_expression[-1] == '(':
        #     algebraic_expression += word
    # algebraic_expression = [alg for alg in algebraic_expression if "<<" in alg]####去除svamp数据集自带公式标注<<的影响
    # print("提取的代数式:", algebraic_expression)
    return algebraic_expression

####isnumeric不支持中间有逗号和负号，重新定义一个函数，浮点数
def isnumber(numberstr):
    try:
        if numberstr[-1] in [',', '.']:###去尾部符号
            numberstr = numberstr[:-1]

        if ',' in numberstr:####去数字间符号
            numberstr = re.sub(',', '', numberstr)

        if numberstr[-1] == '%':
            numberstr = numberstr[:-1]

        return re.sub(r'\.', '', numberstr).isnumeric()###消除点号
    except:
        return False

###消除不符合数字规范的格式
def tonumber(numberstr):
    if numberstr[-1] in [',', '.']:###去尾部符号
        numberstr = numberstr[:-1]

    if ',' in numberstr:####去数字间符号
        numberstr = re.sub(',', '', numberstr)
    return numberstr

###对齐问答中的数字，要格式相同，例如带%，小数等。
def align_num(question, formu):
    # question = re.sub(r'^a-zA-z0-9', ' ', )
    ornums = [tonumber(wd) for wd in re.split(r'[\+\-\*\/\=<>\s]+', question) if isnumber(wd)]###规范问题中的number

    anums = [wd for wd in re.split(r'[\+\-\*\/\=<>\s]+', formu) if isnumber(wd)]####先不规范答案中的number
    ####答案中的错误number格式的字典
    try:
        newnums = {}
        for number in anums:
            for ornum in ornums:
                if ornum[-1] == '%' and ornum != '100%':###原问题是百分号
                    if float(tonumber(number)) == float(ornum[:-1])/100.0:
                        newnums[str(number)] = ornum
                elif float(tonumber(number)) == float(ornum) and str(number) != str(ornum):
                    newnums[str(number)] = ornum
                    break
        # print('check dict ', newnums, '\n', ornums, '\n', anums)
        ###替换答案中的不规范数字
        if len(newnums) > 0:
            for key, val in newnums.items():
                print('before', formu)
                # formu = re.sub(key, val, formu)###特殊符号会出错
                formu = formu.replace(key, val)
                print('after ', formu, 'key value ', key, ' ', val)
    except:
        print('skip alignment ', anums)
    return formu

####乘法规范化(.20)或者(.3)，转换为*0.20或者*0.3的形式
def mul_mon_prep(text):
    pattern = r'\((\.\d+)\)'
    replacement = r'*0\1'###\1定位插入位置
    ans = re.sub(pattern, replacement, text)
    ###120*.5=6修改为 0.5
    ans = re.sub(r'([\*\/\+\-\=\s])\.(\d+)', r'\1 0.\2', ans)  ####（）表示捕获组，第二捕获组定位插入
    ###'600 x 2 / 3 = 400' x修改为 *
    ans = re.sub(r'(\d+\s?)x(\s?\d+)', r'\1 * \2', ans)
    return ans

###从svamp train json解答步骤中找出<<(.*)>>公式
def get_formu(text):
    pattern = r'<<(.*)>>'#### 贪婪匹配.*或.+或.？，找到最长匹配，非贪婪.*？找到最短匹配，括号()则可以排除前后符号，

    formls = re.findall(pattern, text)

    # formls = [alg_nor(form) for form in formls]
    return formls

####找到逻辑相似的样本,从step1得到的forminput中检索,
# formu是从text中提取的规划化formu，只筛选相似度大于threshold的参考样本，如果将threshold设为0，则选8个最相近的作为参考
def get_rag(RAG_samples,formu, threshold=0.9):
    # with open(forminput, 'r', encoding='utf8') as fi:####read tsv or txt
    #     lines = fi.readlines()
    preds = []

    for t in RAG_samples:
        candi_forms = t['formula']

        if len(candi_forms) == 0:
            print('no rag')
            preds.append(0.0)
        else:
            prob = get_log_sim(candi_forms, formu)
            preds.append(prob)
    print('check similary', preds[1:-1:30])
    simidx = np.argsort(preds)[-8:]####
    print('similarity score of logic similar examples ', [preds[idx] for idx in simidx])

    qu_ans = ''####参考样本
    for idx in simidx:
        if preds[idx]>threshold:
            cand = RAG_samples[idx]
            if not isnumber(str(cand['answer'])):
                qu_ans += 'Question: '+cand['question']+'\nAnswer:'+str(cand['answer'])+'\n'
            else:
                qu_ans += 'Question: ' + cand['question'] + '\nAnswer:' + cand['Equation'] + '\n'
    # exit()
    return qu_ans

##语义结合公式筛选相似样本，只筛选相似度大于threshold的参考样本，如果将threshold设为0，则选8个最相近的作为参考
def semlog_rag(RAG_samples,sample, sel_num=3):
    # with open(forminput, 'r', encoding='utf8') as fi:####read tsv or txt
    #     lines = fi.readlines()
    preds = []

    for t in RAG_samples:
        candi_forms = t['formula']
        if len(candi_forms) == 0:
            print('no rag')
            preds.append(0.0)
        else:
            prob = 0.7*get_log_sim(candi_forms, sample['formula']) + 0.3*ques_similarity(t['question'], sample['question'], sent_model)
            preds.append(prob)

    simidx = list(np.argsort(preds)[-sel_num:])####
    simidx.reverse()###相似度高在前
    # print('similarity score of logic similar examples ', [preds[idx] for idx in simidx])

    qu_ans = ''####参考样本
    for idx in simidx:
        cand = RAG_samples[idx]
        qu_ans += 'Question: ' + cand['question']+'\n'
        qu_ans += 'Right answer: ' + re.sub('\n', '', cand['positive']) +'\n'
        qu_ans += 'Wrong answer: ' + re.sub('\n', '', cand['negtive']) + '\n'
        if 'Explanation' in cand.keys():
            qu_ans += 'Explanation: ' + cand['Explanation'] + '\n'
        qu_ans += '|EOS|\n'
    # exit()
    return qu_ans

###数组去重复
def dupli_re(list_case):
    re = []
    for x in list_case:
        if x in re:
            continue
        else:
            re.append(x)
    return re

###step1提取推理步骤中的公式，并保存
def extformula_gsm8k(infile='data/gsm8k_main_train.json',outpath='data/gsm8k_train_formu.json'):

    with open(infile, 'r', encoding='utf8') as fi:####read tsv or txt
        lines = fi.readlines()
    with open(outpath, 'w', encoding='utf8') as fout:
        for idx, data in enumerate(lines):
            data = json.loads(data)
            ent = data['question']
            ans = data['answer']

            ###乘法规范化(.20)，转换为*0.20 120*.5=6修改为 0.5 '600 x 2 / 3 = 400' x修改为 *
            ans = mul_mon_prep(ans)###乘法规范
            # if '$' in ans:
            ans = re.sub(r'\$', '', ans)  ###去除货币符号
            ans = re.sub(r':', r': ', ans)  ###：后增加空格
            ans = re.sub(r'\%\.', r'\% \.', ans)###%.尾部符号非%问题

            cd_formuls = dupli_re(extra_alg2(ans))#######消除重复
            print('extra ', cd_formuls, ans)
            formuls = []
            for form in cd_formuls:
                pattern = r'<<(.*)>>'  #### 贪婪匹配.*或.+或.？，找到最长匹配，非贪婪.*？找到最短匹配，括号()则可以排除前后符号，
                nform = re.findall(pattern, form)
                if len(nform) > 0:
                    formuls.append(nform[0])
                else:
                    formuls.append(form)
            # ans = '|'.join(formuls)
            # formuls = get_formu(ans)
            # formuls = [form for form in formuls if not (len(form) == 3 and '=' in form)]  ###去除x=x此类结论步骤
            print('extra form', formuls)

            formuls = [align_num(ent, formu) for formu in formuls]  ###对齐value
            # formuls = re.split(' ', formuls)
            print('after align ', formuls)

            ####规范化、归一化
            formuls = [alg_nor(form) for form in formuls]
            formuls = [form for form in formuls if not (len(form) == 3 and '=' in form)]###去除x=x此类结论步骤
            print('the ', idx, 'th question ', 'check get formu ', formuls)##, '\n ', ans)

            data['formula'] = formuls
            if len(formuls) == 0:
                print('the ', idx, 'th question ', '\n ', ans)

            fout.write(json.dumps(data) + "\n")
            # json.dump(data, fout, indent=4, separators=(", ", ": "), sort_keys=True)###循环写入，有缩进,但无换行符，读取有风险

###在公式字符串中定位并条件替换某值
def loc_sub_str(formu, value, cform):
    formu_parts = re.split(r'([\+\-\*\\\/\(\)])', formu)###含有分割符的分割
    print('check loc sub', formu, formu_parts, value,cform)
    if value in formu_parts:
        if '-' in cform or '+' in cform:
            formu_parts[formu_parts.index(value)] = '('+cform+')'
        else:
            formu_parts[formu_parts.index(value)] = cform###替换
    print('after loc sub ', ''.join(formu_parts))
    return ''.join(formu_parts)

###step1提取推理步骤中的公式，并将多个步骤合成综合表达式
def ext_comb_form_gsm8k(infile='data/gsm8k_main_train.json',outpath='data/gsm8k_train_formu1.json'):

    with open(infile, 'r', encoding='utf8') as fi:####read tsv or txt
        lines = fi.readlines()
    with open(outpath, 'w', encoding='utf8') as fout:
        for idx, data in enumerate(lines):
            data = json.loads(data)
            ent = data['question']
            ans = data['answer']

            ###乘法规范化(.20)，转换为*0.20 120*.5=6修改为 0.5 '600 x 2 / 3 = 400' x修改为 *
            ans = mul_mon_prep(ans)###乘法规范
            # if '$' in ans:
            ans = re.sub(r'\$', '', ans)  ###去除货币符号
            ans = re.sub(r':', r': ', ans)  ###：后增加空格
            ans = re.sub(r'\%\.', r'\% \.', ans)###%.尾部符号非%问题

            cd_formuls = dupli_re(extra_alg2(ans))#######消除重复
            # print('extra ', cd_formuls, ans)
            formuls = []
            for form in cd_formuls:
                pattern = r'<<(.*)>>'  #### 贪婪匹配.*或.+或.？，找到最长匹配，非贪婪.*？找到最短匹配，括号()则可以排除前后符号，
                nform = re.findall(pattern, form)
                if len(nform) > 0:
                    formuls.append(nform[0])
                else:
                    formuls.append(form)

            formuls = [form for form in formuls if any(x in form for x in ['+', '-', '*', '/', '\\'])]  ###去除x=x此类结论步骤
            print('extra form', formuls)

            if len(formuls)<1:###跳过不规范样本
                continue

            formuls = [align_num(ent, formu) for formu in formuls]  ###对齐value
            # formuls = re.split(' ', formuls)
            # print('after align ', formuls)

            ####规范化、归一化
            # formuls = [alg_nor(form) for form in formuls]
            # formuls = [form for form in formuls if not (len(form) == 3 and '=' in form)]###去除x=x此类结论步骤
            print('the ', idx, 'th question ', 'check get formu ', formuls)##, '\n ', ans)

            finform_parts = formuls[-1].split('=')
            if any(x in finform_parts[0] for x in ['+', '-', '*', '/', '\\']):
                finform = finform_parts[0]
            else:
                finform = finform_parts[1]
            ###迭代替换
            if len(formuls)>1:
                preforms = formuls[:-1]
                preforms.reverse()###逆向

                for formu in preforms:
                    if not '=' in formu:
                        continue
                    parts = formu.split('=')
                    if any(x in parts[1] for x in ['+', '-', '*', '/', '\\']):
                        value = parts[0]
                        cform = parts[1]
                    else:
                        value = parts[1]###单值
                        cform = parts[0]###表达式
                    finform = loc_sub_str(finform, value, cform)###定位并替换

            print('combined formula', finform)
            data['formula'] = alg_nor(finform)
            # print('combined formula normalized ', data['formula'])
            fout.write(json.dumps(data) + "\n")
            # json.dump(data, fout, indent=4, separators=(", ", ": "), sort_keys=True)###循环写入，有缩进,但无换行符，读取有风险


###测试样本，输入为样本的id序号
def test_rag_answer(args):
    tests = []
    with open(args.infile, 'r', encoding='utf-8') as fi:
        for lin in fi:
            tests.append(json.loads(lin))
    if len(tests)>100:
        idxs = range(100)
    else:
        idxs = range(len(tests))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_4bit=True,  ##torch_dtype=torch.int8,
                                                 trust_remote_code=True)
    model = model.eval()
    gen_kwargs = {"max_new_tokens": 400, "top_p": 0.95, "temperature": 0.1, "top_k": 30,
                  "do_sample": True, 'repetition_penalty': 1.15}  ###不用一次返回多个答案
    gen_kwargs['eos_token_id'] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    with open(args.RAG_path, 'r', encoding='utf-8') as fi:
        RAG_samples = [json.loads(data) for data in fi.readlines()]

    ####output路径
    output_data_path = args.outfile###test_data_path.replace('.tsv', '.json')
    generated_responses = []

    if os.path.exists(output_data_path):
        with open(output_data_path, 'r', encoding='utf8') as fin:
            generated_responses = fin.readlines()
        print(f"Loaded {len(generated_responses)} generated responses")

    newdata = []
    with open(output_data_path, 'w', encoding='utf8') as fout:
        for idx in idxs:
            data = tests[idx]
            ent = data['question']
            ans = data['answer']
            prompt = "Let's first understand the problem and devise a plan to solve the problem.Then, carry out the plan to solve the problem step by step"
            rag_rst = semlog_rag(RAG_samples, data, args.shot_num)
            rag_rst = 'The following are examples of math problems and their solutions, which have wrong and right answers. ' \
                      'Please refer to the right answer to solve the new problem,and also avoid mistakes in the wrong answers.\n' \
                       + rag_rst + '##########\nFollowing is the new problem to solve. Question: '

            prompt = rag_rst + ent + prompt  #####RAGenhance，找出相关推理问题

            inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

            res = []
            for _ in range(5):  ####给出5个结果,self correction 耗时太长，不比较

                generate_ids = model.generate(**inputs, **gen_kwargs)
                if not any(end_token in generate_ids[0] for end_token in gen_kwargs['eos_token_id']):
                    print('entering re generate data ###############')
                    gen_kwargs['max_new_tokens'] = gen_kwargs['max_new_tokens'] + 128  ###生成长度
                    generate_ids = model.generate(**inputs, **gen_kwargs)
                for gen_out in generate_ids:
                    response = tokenizer.decode(gen_out, skip_special_tokens=True)

                    response = response.replace(prompt, "").strip()  ####消去已知条件
                    # response = re.findall(r'(##\nFollowing is the new problem).+$', response)[0]
                    ####检查答案的完整性，剔除不完整答案。

                    print('responese ', response, '\n label and answer ', data['answer'], extract_answer('gsm8k', response))
                    res.append(response)
            data['predict'] = res
            generated_responses.append(data)
            fout.write(json.dumps(data)+'\n')

###两步法解决问题，先plan 后solve
def test_rag_2step(args):
    print()

import openai
def get_chatgpt_reply(question):
    openai.api_type = "azure"
    openai.api_base = "your azure base"  ##gpt35-dk
    openai.api_version = '2023-03-15-preview'
    openai.api_key = 'your api'
    # question = '规划去杭州的旅游，4天时间'
    try:
        chat = openai.ChatCompletion.create(engine="gpt35-turbo",  ###azure
                                            messages=[{"role": "user", "content": question}],
                                            max_tokens=400,
                                            temperature=0.8)
    # chat = openai.ChatCompletion.create(engine="gpt35instruct",  ###azure
    #                                     messages=[{"role": "user", "content": question}],
    #                                     max_tokens=400,
    #                                     temperature=0.8)
    except:
        time.sleep(30)
        chat = openai.ChatCompletion.create(engine="gpt35-dk",  ###azure
                                            messages=[{"role": "user", "content": question}],
                                            max_tokens=400,
                                            temperature=0.8)
    reply = chat.choices[0].message.content
    print('check chatgpt result ', reply)

    return reply

def test_chatgpt(args):
    tests = []
    with open(args.infile, 'r', encoding='utf-8') as fi:
        for lin in fi:
            tests.append(json.loads(lin))
    if len(tests) > 100:
        idxs = range(100)
    else:
        idxs = range(len(tests))

    with open(args.RAG_path, 'r', encoding='utf-8') as fi:
        RAG_samples = [json.loads(data) for data in fi.readlines()]

    ####output路径
    output_data_path = 'data/gsm8k_gpt_shot{}.json'.format(
        args.shot_num)  ###test_data_path.replace('.tsv', '.json')
    generated_responses = []

    if os.path.exists(output_data_path):
        with open(output_data_path, 'r', encoding='utf8') as fin:
            generated_responses = fin.readlines()
        print(f"Loaded {len(generated_responses)} generated responses")

    with open(output_data_path, 'a', encoding='utf8') as fout:
        for idx in idxs:
            if idx < len(generated_responses):####跳过
                continue

            data = tests[idx]
            ent = data['question']
            prompt = " Let's first understand the problem and devise a plan to solve the problem. " \
             "Then, let's carry out the plan to solve the problem step by step."

            rag_rst = semlog_rag(RAG_samples, data, args.shot_num)
            rag_rst = 'The following are examples of math problems and their solutions, which have wrong and right answers. Afterwards, you will be given a new problem to solve. ' \
                      + rag_rst + 'The examples are similar to the new problem, so you should refer to the right answer to solve the new problem,' \
                                  'and also avoid mistakes in the wrong answers.\n##########\nFollowing is the new problem to solve. Question: '

            prompt = rag_rst + ent + prompt  #####RAGenhance，找出相关推理问题

            res = []
            for _ in range(5):  ####给出5个结果,self correction 耗时太长，不比较
                response = get_chatgpt_reply(prompt)
                print('responese ', response, '\n label and answer ', data['answer'],
                      extract_answer('gsm8k', response))
                res.append(response)
            data['predict'] = res
            generated_responses.append(data)
            fout.write(json.dumps(data) + '\n')

####训练集太多，先用检索的方式，找到需要进行推理的样本
def collect_similar_samples(infile='data/gsm8k_test_formu1.json',infile2='data/gsm8k_train_formu1.json', outpath='data/gsm8k_train_refersamples.json'):
    tests = []
    trains = []
    with open(infile, 'r', encoding='utf-8') as fi:
        for lin in fi:
            tests.append(json.loads(lin))
    with open(infile2, 'r', encoding='utf-8') as fi:
        for lin in fi:
            trains.append(json.loads(lin))
    idxs = []
    for sample in tests[:100]:
        preds = []####相似度
        for t in trains:
            candi_forms = [alg_nor(t['formula'])]
            sam_for = [alg_nor(sample['formula'])]
            if len(candi_forms) == 0:
                print('no rag')
                preds.append(0.0)
            else:
                prob = 0.7 * get_log_sim(candi_forms, sam_for) #+ 0.3 * ques_similarity(t['question'], sample['question'], sent_model)
                preds.append(prob)

        simidx = list(np.argsort(preds)[-9:])  ####最多9个相似样本
        idxs.extend(simidx)
    sel_inds = list(set(idxs))

    proced = []
    with open('data/gsm8k_posneg1.json', 'r', encoding='utf-8') as fi:
        for lin in fi:
            proced.append(json.loads(lin)['question'])

    with open(outpath, 'w', encoding='utf8') as fout:
        for idx in sel_inds:
            data = trains[idx]
            if data['question'] in proced:###跳过
                continue
            fout.write(json.dumps(data) + '\n')


if __name__ == '__main__':

    # explain_wrong(args)
    ##test_chatgpt(args)
    test_rag_answer(args)
    # test_2rag_comb(args)
    # test_classify100(args, 100)



