import re
from typing import Union
import json
from collections import Counter
def extract_finance(text):
    pattern = '-?\d+\.?\d*%?'
    pred = re.findall(pattern, text)
    if pred:
        if '%' == pred[-1][-1]:
            pred_answer = eval(pred[-1][:-1] + '/100')
        else:
            pred_answer = float(pred[-1])
        return pred_answer
    pattern = 'yes|no'
    pred = re.findall(pattern, text)
    if pred:
        return pred[-1]
    return None

###原始求解
def extract_answer(dataset, text):
    # dataset = args.dataset.lower()
    if dataset in ["svamp", "gsm8k", "multiarith", "addsub", "singleeq"]:
        pred_answer = extract_number(text)
    elif dataset == "commonsenseqa":
        pred = text.strip()
        pred = re.sub("\(|\)|\:|\.|\,", "", pred)
        pred = pred.split()
        pred_answer = [i for i in pred if i in ('A|B|C|D|E')][-1]
        # pred_answer = re.findall(r'A|B|C|D|E', pred)[0]
        return pred_answer
    elif dataset == "aqua":
        pred = text.strip()
        pred_answer = re.findall(r'A|B|C|D|E', pred)[0]
        return pred_answer
    elif dataset == "strategyqa" or dataset == 'coin_flip':
        pred = text.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split()
        pred_answer = [i for i in pred if i in ("yes", "no")][-1]
        return pred_answer
    elif dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s", "", text)
        pred_answer = pred
        return pred_answer
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

###优化求解，最后一句答案中有两个数时
def extract_answer_opt(question, answer):
    # print('check answer before ', answer)
    answer = cut_right_answer(answer)
    # print('check answer after cutt ', answer)
    answer = answer.strip()
    answers = re.split(r'\n', answer)  ###去掉换行
    # print('check answer', answer[-1])
    # lines = re.split(r'[\;\。]', answer)  ###按标点分割
    # print('check answer', answer)
    if len(answers[-1]) > 40:###字符串长度
        answer = answers[-1]
    # else:
        # answer = answers[0]
        # print('check answers ', answers)
    answer = answer.replace(',', '')  ####去掉数字中的，
    # print('check answer', answer)
    pred_num = [s for s in re.findall(r'-?\d+\.?\d*', answer)]####
    if len(pred_num) == 0:
        return None
    if len(pred_num) > 1:
        ques_num = [s for s in re.findall(r'-?\d+\.?\d*', question)]
        for q in ques_num:
            if q in pred_num:
                pred_num.pop(pred_num.index(q))
        if len(pred_num) == 0:
            # print('check answer #####', answer, question)
            pred_num = [None]
    pred_answer = pred_num[-1]
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def get_precision(gt_ans: float) -> int:
    precision = 5
    if '.' in str(gt_ans):
        precision = len(str(gt_ans).split('.')[-1])
    return precision

####
def extract_number(text: str) -> Union[float, None]:
    text = text.replace(',', '')####去掉数字中的，

    pred = [s for s in re.findall(r'-?\d+\.?\d*', text)]####最后一个数值，不太准确
    if pred:
        pred_answer = float(pred[-1])
    else:
        pred_answer = None
    return pred_answer

###交叉检验
def cross_check(jsonfile1, jsonfile2):
    with open(jsonfile1, 'r', encoding='utf8') as fin:
        lines1 = fin.readlines()
    with open(jsonfile2, 'r', encoding='utf8') as fin:
        lines2 = fin.readlines()
    corrects = 0.0
    counter = 0.0
    uncertain = 0.0
    for line1, line2 in zip(lines1, lines2):
        # if counter==87.0:
        #     break
        counter += 1.0
        # print(line)
        data1 = json.loads(line1)
        data2 = json.loads(line2)
        lab_txt1 = data1['answer']
        if 'predict' in data1.keys():
            prd_txt1 = data1['predict']
        else:
            prd_txt1 = data1['newpred']
        if 'predict' in data2.keys():
            prd_txt2 = data2['predict']
        else:
            prd_txt2 = data2['newpred']

        lab_txt2 = data2['answer']
        if 'gsm8k' in jsonfile1:
            label1 = extract_answer('gsm8k', lab_txt1)
            label2 = extract_answer('gsm8k', lab_txt2)
            pred1 = [extract_answer('gsm8k', tt) for tt in prd_txt1]  ###答案列表
            pred2 = [extract_answer('gsm8k', tt) for tt in prd_txt2]  ###答案列表
        else:
            label1 = float(lab_txt1)
            label2 = float(lab_txt2)
            pred1 = [extract_answer('svamp', tt) for tt in prd_txt1]
            pred2 = [extract_answer('svamp', tt) for tt in prd_txt2]

        if label1 in pred1 or (label1 in pred2):
            corrects += 1.0
            if Counter(pred1+pred2).most_common(1)[0][0] != label1:
                uncertain += 1.0
        else:
            print(counter, ' label', label1, ' ', label2, '\npredt', pred1, pred2, '\n', prd_txt2)

        # print('check label pred', label, pred, Counter(pred).most_common(1)[0][0], '\n', lab_txt, '\n', prd_txt[0])
    print('accuracy', corrects / counter, 'totalnum ', counter, 'uncertain', uncertain/counter)
###计算精度
def cal_acc_main(datapath):
    negidx = []
    corrects = 0.0
    uncertain = 0.0
    counter = 0.0
    correct_wo_sc = 0.0
    with open(datapath, 'r', encoding='utf8') as fin:
        lines = fin.readlines()
        for line in lines:
            counter += 1.0
            # print(line)
            data = json.loads(line)
            if 'answer' in data.keys():
                lab_txt = data['answer']
            else:
                lab_txt = data['Answer']
            if 'predict' in data.keys():
                prd_txt = data['predict']
            else:
                prd_txt = data['newpred']

            if 'gsm8k' in datapath:
                label = extract_answer('gsm8k', lab_txt)
                pred = [extract_answer('gsm8k', tt) for tt in prd_txt]  ###答案列表
            else:
                label = float(lab_txt)
                pred = [extract_answer('svamp', tt) for tt in prd_txt]

            if label == pred[0]:  ####
                correct_wo_sc += 1.0
            if label in pred:
                corrects += 1.0
                if Counter(pred).most_common(1)[0][0] != label:####高概率选出
                    uncertain += 1.0
                    # print('label', label, ' indx ', int(counter), '\npredt', pred, '\n', prd_txt)
            else:
                # print('label', label, ' indx ', int(counter), '\npredt', pred, '\n', prd_txt)
                # exit()
                negidx.append(int(counter-1))
            # print('check label pred', label, pred, Counter(pred).most_common(1)[0][0], '\n', lab_txt, '\n', prd_txt[0])
        print('accuracywosc', correct_wo_sc / counter, 'accuracy', (corrects-uncertain) / counter, 'uncert',
              corrects / counter, 'totalnum ', counter)
        return negidx

def cal_acc_main2(datapath):
    negidx = []
    corrects = 0.0
    uncertain = 0.0
    counter = 0.0
    correct_wo_sc = 0.0
    with open(datapath, 'r', encoding='utf8') as fin:
        lines = fin.readlines()
        for line in lines:
            counter += 1.0
            if counter>100:
                break
            # print(line)
            data = json.loads(line)
            if 'answer' in data.keys():
                lab_txt = data['answer']
            else:
                lab_txt = data['Answer']
            if 'predict' in data.keys():
                prd_txt = data['predict']
            else:
                prd_txt = data['newpred']

            if 'gsm8k' in datapath:
                label = extract_answer('gsm8k', lab_txt)
                pred = [extract_answer_opt(data['question'], tt) for tt in prd_txt[:5]]  ###答案列表
            else:
                label = float(lab_txt)
                pred = [extract_answer_opt(data['Question'], tt) for tt in prd_txt]
            pred = [p for p in pred if p!=None]
            if len(pred)==0:
                continue
            if label == pred[0]:  ####
                correct_wo_sc += 1.0
            if label in pred:
                corrects += 1.0
                if Counter(pred).most_common(1)[0][0] != label:####高概率选出
                    uncertain += 1.0
                    print('label', label, ' indx ', int(counter), '\npredt', pred, '\n##############', prd_txt[0])
            else:
                print('label', label, ' indx ', int(counter), '\npredt', pred, prd_txt[0])
                # exit()
                negidx.append(int(counter-1))
            # print('check label pred', label, pred, Counter(pred).most_common(1)[0][0], '\n', lab_txt, '\n', prd_txt[0])
        print('accuracywosc', correct_wo_sc / counter, 'accuracy', (corrects-uncertain) / counter, 'uncert',
              corrects / counter, 'totalnum ', counter)
        return negidx


###表达式规范化，数值或字符转# 替换匹配到的数字为单个"@"字符
def alg_nor(input_str):
    ###分别替换公式中的单词或数字
    pattern = r'\d+'####r'-?\d+\.?\d+'  ##数字只是整数，####保留小数，负数，百分数等，例如-100，-2.3，1.11等方便对比
    result = re.sub(pattern, '@', input_str)
    pattern = r'[a-zA-Z]+'##
    result = re.sub(pattern, '@', result)

    # ###不规范小数点，保留
    # result = re.sub(r'\.@', r'@', result)###
    ###不规范推理
    result = re.sub(r'=.*=', r'=', result)
    ###不规范乘法符号，x(x)
    result = re.sub(r'@\(', r'@*(', result)
    result = re.sub(r'@@', r'@*@', result)
    return result

###判断llm答案的完整性，比如最后一句完整结束，并且有so，therefor等结论词
def check_answer_over(answer):
    # answer = re.sub(r'\n', ' ', answer)###去掉换行
    lines = re.split(r'\n', answer)###按标点分割
    over_ws = ['so', 'So', 'therefor', 'Therefore', 'answer', 'Answer', 'boxed', 'Boxed', 'Thus', 'conclusion']
    lastline = lines[-1]
    # print('check answer completeness', lastline)  ##integrity
    if any(wd in lastline for wd in over_ws):
        if lastline[-1]=='.':
            return True
        else:
            return False
    else:

        return False
###找出完整答案，input为data问题，preds多个预测结果，mostrank为正负预测统计排名, flag为解答的正负标志的data
def get_integ_answer(data, preds, label):
    # sids = [k for k, v in Counter(preds).most_common(1)]  ###错误答案value
    for sid, pred in enumerate(preds):
        if pred == label and check_answer_over(data['predict'][preds.index(sid)]):
            return data['predict'][preds.index(sid)]
    ##没发现完整答案，则###记录第1个
    print('no complete answer')
    return data['predict'][preds.index(label)]


####找出错误，并进行总结，人工去做，难总结
def explain_wrong_steps():
    prompt = 'find the error steps and summarize the reasons'
    print()


def log_postpoc(jsonfile, outfile):
    datas = []
    with open(jsonfile, 'r', encoding='utf8') as fi:
        for lin in fi:
            print(lin)
            data = json.loads(lin)
            datas.append(data)
    postpros = []
    for line in datas:
        reslts = line['predict']
        npred = re.split(r'label and answer',reslts[0])
        ans = npred[1].strip().split(' ')[0]
        line['Answer'] = ans
        postpros.append(line)

    with open(outfile, 'w') as fo:
        for exam in postpros:
            fo.write(json.dumps(exam)+'\n')

import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
def draw_com():
    mistr_shots_5guess_lat = [0.74, 0.698, 0.755, 0.734, 0.724, 0.772]###logic shot
    mistr_shots_5guess_acc = [0.61, 0.571, 0.592, 0.622, 0.643, 0.663]
    mistr_shots_10guess_lat = [0.82, 0.714, 0.755, 0.806, 0.796, 0.816]
    mistr_shots_10guess_acc = [0.57, 0.592, 0.613, 0.664, 0.643, 0.724]

    gsm8k_shots_10guess_lat = [0.57, 0.58, 0.59, 0.61, 0.63, 0.60]###logic shot
    gsm8k_shots_10guess_acc = [0.38, 0.39, 0.35, 0.42, 0.46, 0.40]
    gsm8k_shots_5guess_lat = [0.55, 0.52, 0.54, 0.56, 0.59, 0.56]
    gsm8k_shots_5guess_acc = [0.42, 0.37, 0.37, 0.42, 0.44, 0.41]

    blender_10guess_lat = [0.842]
    blender_10guess_acc = [0.703]
    mistr_sc_5guess_shots_lat = [0.78, 0.82]###fix hard 5guess and 10 guess
    mistr_sc_5guess_shots_acc = [0.55, 0.57]
    mistr_sc_10guess_shots_lat = [0.85, 0.88]###5guess and 10 guess
    mistr_sc_10guess_shots_acc = [0.67, 0.68]

    plt.style.use('science')
# # plt.style.use(['science', 'no-latex'])####plt.style.use(['science', 'high-contrast','no-latex']) plt.style.use(['ieee', 'no-latex']) plt.rcParams['font.family'] = "Times New Roman"
# def fid_plot():
    fig, axs = plt.subplots(1, 1, sharey=False, figsize=(6, 4))
    plt.xlim((0, 9))
    plt.ylim((0.3, 0.9))
    x = [0, 1, 3, 5, 7, 9]

    sns.lineplot(x=x, y=mistr_shots_5guess_acc, marker='*', ax=axs, legend='auto', label='guess-5')
    # sns.lineplot(x=[8], y=mistr_sc_5guess_shots_acc[:1], marker='*', ax=axs, legend='auto', label='hard-5')
    sns.lineplot(x=x, y=mistr_shots_10guess_acc, marker='*', ax=axs, legend='auto', label='guess-10')
    # sns.lineplot(x=[8], y=mistr_sc_10guess_shots_acc[:1], marker='*', ax=axs, legend='auto', label='hard-10')

    sns.lineplot(x=x, y=mistr_shots_5guess_lat, ax=axs, legend='auto', label='latent-5')
    # sns.lineplot(x=[7.3, 7.8], y=mistr_sc_5guess_shots_lat, ax=axs, legend='auto', label='fix-lat-5')
    sns.lineplot(x=x, y=mistr_shots_10guess_lat, ax=axs, legend='auto', label='latent-10')
    # sns.lineplot(x=[7.3, 7.8], y=mistr_sc_10guess_shots_lat, ax=axs, legend='auto', label='fix-lat-10')
    ##finetune


    axs.set_xlabel('Shot numbers')
    axs.set_ylabel('Accuracy')
    plt.savefig('algcomacc.png')


def draw_gsm8k():
    gsm8k_shots_10guess_lat = [0.62, 0.63, 0.68, 0.71, 0.71, 0.68]###logic shot
    gsm8k_shots_10guess_acc = [0.43, 0.46, 0.44, 0.53, 0.56, 0.48]
    gsm8k_shots_5guess_lat = [0.55, 0.52, 0.57, 0.56, 0.59, 0.56]
    gsm8k_shots_5guess_acc = [0.42, 0.37, 0.37, 0.42, 0.44, 0.41]
    plt.style.use('science')
# # plt.style.use(['science', 'no-latex'])####plt.style.use(['science', 'high-contrast','no-latex']) plt.style.use(['ieee', 'no-latex']) plt.rcParams['font.family'] = "Times New Roman"
# def fid_plot():
    fig, axs = plt.subplots(1, 1, sharey=False, figsize=(6, 4))
    plt.xlim((0, 9))
    plt.ylim((0.15, 0.75))
    x = [0, 1, 3, 5, 7, 9]

    sns.lineplot(x=x, y=gsm8k_shots_5guess_acc, marker='*', ax=axs, legend='auto', label='guess-5')
    sns.lineplot(x=x, y=gsm8k_shots_10guess_acc, marker='*', ax=axs, legend='auto', label='guess-10')
    sns.lineplot(x=x, y=gsm8k_shots_5guess_lat, ax=axs, legend='auto', label='latent-5')
    sns.lineplot(x=x, y=gsm8k_shots_10guess_lat, ax=axs, legend='auto', label='latent-10')

    axs.set_xlabel('Shot numbers')
    axs.set_ylabel('Accuracy')
    plt.savefig('algcom_gsm.png')
####竟然给出了wrong answer，删除掉
def cut_right_answer(line):
    return re.sub(r'(Wrong answer:)[\s\S]+', '', line)####wronganswer后所有
if __name__=='__main__':

    # a = "To the initial 2 pounds of jelly beans, he added enough brownies to cause the weight to triple, bringing the weight to 2*3=<<2*3=6>>6 pounds.\nNext, he added another 2 pounds of jelly beans, bringing the weight to 6+2=<<6+2=8>>8 pounds.\nAnd finally, he added enough gummy worms to double the weight once again, to a final weight of 8*2=<<8*2=16>>16 pounds.\n#### 16"
    # b = re.sub(r'<<.*>>', '', a)
    # print(b)
    # exit()
    # draw_com()
    # draw_gsm8k()

    g5_jsonls = ['data/svamp-g5/mistral_shot1.json', 'data/svamp-g5/mistral_shot3.json',
              'data/svamp-g5/mistral_shot5.json', 'data/svamp-g5/mistral_shot7.json', 'data/svamp-g5/svamp_gpt_shot5.json','data/svamp-g5/mistral_shot9.json']

    jsonls = ['data/svamp-10/svamp_mistral_val_sc_shothard.json', 'data/svamp-10/svamp_mistral_val_sc_shotfix.json',
              'data/svamp-g5/mistral_shotfix.json', 'data/svamp-g5/mistral_shothard.json']
    g10_jsonls = ['data/svamp-10/svamp_guess10_shot1.json', 'data/svamp-10/svamp_guess10_shot3.json',
    'data/svamp-10/svamp_guess10_shot7.json','data/svamp-10/svamp_guess10_shot5.json', 'data/svamp-10/svamp_guess10_shot9_p.json']
    aritho_jsonls = ['data/svamp-g5/aritho5.json', 'data/svamp-g5/aritho5fix.json',
                    'data/svamp-g5/aritho5hard.json', 'data/svamp-g5/arithog5_shot9.json']
    # for jsonfile in g5_jsonls:
    #     print(jsonfile, ' ', cal_acc_main2(jsonfile))
    cal_acc_main2('data/gsm8k_g10log.json')
    # # cal_acc_main2('data/svamp-g5/svamp_gpt_shot5.json')
    exit()

    # jsonls = ['data/gsm8k-guess5/gsm8k_main_test_cot.json', 'data/gsm8k-guess5/gsm8k_main_test_sc.json', 'data/gsm8k-guess5/gsm8k_main_test_ps.json',
    #           'data/gsm8k-guess5/gsm8k_main_test_aritho.json', 'data/gsm8k-guess5/gsm8k_main_test_aritho_sc.json', 'data/gsm8k-guess5/gsm8k_main_test_aritho_ps.json',
    #           'data/gsm8k-guess5/gsm8k_llama3_tr1_sc.json']
    g10jsonls = ['data/gsm8k-10/gsm8k_guess10_shot1.json', 'data/gsm8k-10/gsm8k_guess10_shot3.json', 'data/gsm8k-10/gsm8k_guess10_shot5.json', 'data/gsm8k-10/gsm8k_guess10_shot7.json', 'data/gsm8k_guess10_shot9.json']
    for jsonfile in g10jsonls:
        print(jsonfile, ' ', cal_acc_main2(jsonfile))
    exit()
    cross_check(jsonls[3], jsonls[2])
    # cross_check(jsonls[1], jsonls[2])
    exit()

