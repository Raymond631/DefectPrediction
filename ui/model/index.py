import tkinter as tk
from tkinter import ttk

from models.ADTree.adtree import ad_tree
# 引入模型
from models.Adaboost.adaboost import adaboost
from models.DT.dt import decision_tree
from models.LR.lrtest import lr
from models.Naive_Bayes.model_nb import naive_bayes
from models.XGboost.xg import xgboost
from models.mlp.mlp_nk import multilayer_perceptron
from models.svm.svm import svm

# 全局变量
data_set = "AEEEM"
# 可供选择的模型
buttons_single = []
buttons_single_texts = ["naive_bayes", "svm", "ADTree", "dt", "lr", "mlp", "adaboost", "xgboost", "random_forest",
                        "knn"]
buttons_combine = []
buttons_combine_texts = ["naive_bayes", "svm", "ADTree", "DT", "LR", "mlp", "adaboost", "xgboost", "random_forest",
                         "knn"]
single = 0
combine = 0
single_model = 'naive_bayes'
models = []
# single(0) or combine(1)
soc = 0


# 提交按钮
def button_data_click():
    global soc
    global models
    global data_set
    global single_model
    folder_path = '../../data/arff/' + data_set
    print(f"数据集: {data_set}")
    # 数据集判断
    bug_label = b'buggy'

    if soc == 1:
        if len(models) > 1:
            print(f"组合模型: {models}")
        else:
            print('模型选择需要大于等于2')
    else:
        print(f"单个模型: {single_model}")
        if single_model == 'naive_bayes':
            naive_bayes(folder_path, bug_label)
        elif single_model == 'svm':
            svm(folder_path, bug_label)
        elif single_model == 'ADTree':
            ad_tree(folder_path, bug_label)
        elif single_model == 'dt':
            decision_tree(folder_path, bug_label)
        elif single_model == 'lr':
            lr(folder_path, bug_label)
        elif single_model == 'mlp':
            multilayer_perceptron(folder_path, bug_label)
        elif single_model == 'adaboost':
            adaboost(folder_path, bug_label)
        elif single_model == 'xgboost':
            xgboost(folder_path, bug_label)
        elif single_model == 'random_forest':
            # random_forest(folder_path, bug_label)
            print("暂无模型")
        elif single_model == 'knn':
            print("暂无模型")

    label_select.config(text="Button Clicked!")


# 组合模型预测的模型选择
def on_checkbox_click():
    global models
    models = []
    for checkbox_var, checkbox_text in zip(buttons_combine, buttons_combine_texts):
        if checkbox_var.get() == 1:
            models.append(checkbox_text)
    print("已选中的选项:", models)


# 数据集选择
def on_radio_button_click():
    global data_set
    data_set = radio_var.get()
    print("已选中的选项:", data_set)


# 单个模型选择
def radio_single_model():
    global single_model
    single_model = radio_single.get()
    print("选中的值为:", single_model)


# 选择组合模型预测
def show_models_combine():
    global soc
    global combine
    global single
    soc = 1
    if single == 1:
        container_single_model_show.pack_forget()
    container_combine_model_show.pack()
    if combine == 0:
        for i in range(len(buttons_combine_texts)):
            check_combines = tk.IntVar()
            buttons_combine.append(check_combines)
            checkbox = tk.Checkbutton(container_combine_model_show, text=buttons_combine_texts[i],
                                      variable=check_combines,
                                      command=on_checkbox_click)
            checkbox.grid(row=i // 5, column=i % 5)
        combine = 1


# 选择单模型预测
def show_models_single():
    global soc
    global combine
    global single
    soc = 0
    if combine == 1:
        container_combine_model_show.pack_forget()
    container_single_model_show.pack()
    if single == 0:
        for i in range(len(buttons_single_texts)):
            radiobutton = ttk.Radiobutton(container_single_model_show, text=buttons_single_texts[i],
                                          variable=radio_single,
                                          value=buttons_single_texts[i], command=radio_single_model)
            radiobutton.grid(row=i // 5, column=i % 5)
            buttons_single.append(radiobutton)
        single = 1


root = tk.Tk()
root.title('缺陷预测模型')
root.geometry('450x300')
label_title = tk.Label(root, text="缺陷预测模型")
label_title.pack(side='top')

# 数据选择
container_data_select = tk.Frame(root)
label_data = tk.Label(root, text="数据选择")
label_data.pack(side=tk.TOP)
radio_var = tk.StringVar(value="AEEEM")
radio_button1 = tk.Radiobutton(container_data_select, text="AEEEM", variable=radio_var, value="AEEEM",
                               command=on_radio_button_click)
radio_button1.pack(side=tk.LEFT)
radio_button2 = tk.Radiobutton(container_data_select, text="MORPH", variable=radio_var, value="MORPH",
                               command=on_radio_button_click)
radio_button2.pack(side=tk.LEFT)
radio_button3 = tk.Radiobutton(container_data_select, text="RELINK", variable=radio_var, value="RELINK",
                               command=on_radio_button_click)
radio_button3.pack(side=tk.LEFT)
radio_button4 = tk.Radiobutton(container_data_select, text="SOFTLAB", variable=radio_var, value="SOFTLAB",
                               command=on_radio_button_click)
radio_button4.pack(side=tk.LEFT)
container_data_select.pack()

# 模型选择
container_model_select = tk.Frame(root)
label_model = tk.Label(root, text="模型选择")
label_model.pack(side=tk.TOP)
container_model_show = tk.Frame(root)
button_models = tk.Button(container_model_show, text="单模型预测", command=show_models_single)
button_models.pack(side=tk.LEFT, padx=10)
button_models_single = tk.Button(container_model_show, text="组合模型预测", command=show_models_combine)
button_models_single.pack(side=tk.LEFT)
container_model_show.pack()
container_single_model_show = tk.Frame(container_model_select)
radio_single = tk.StringVar(value=buttons_single_texts[0])
container_single_model_show.pack()
container_combine_model_show = tk.Frame(container_model_select)
container_combine_model_show.pack()
container_model_select.pack()
button_data = tk.Button(root, text="确认选择", command=button_data_click)
button_data.pack(fill=tk.Y, pady=30)
label_select = tk.Label(root, text="")
label_select.pack(side=tk.BOTTOM)
root.mainloop()
