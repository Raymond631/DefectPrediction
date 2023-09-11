import tkinter as tk

# root_window = tk.Tk()
# # 设置窗口title
# root_window.title('C语言中文网：c.biancheng.net')
# # 设置窗口大小:宽x高,注,此处不能为 "*",必须使用 "x"
# root_window.geometry('450x300')
# # 更改左上角窗口的的icon图标,加载C语言中文网logo标
# # root_window.iconbitmap('C:/Users/Administrator/Desktop/favicon.ico')
# # 设置主窗口的背景颜色,颜色值可以是英文单词，或者颜色值的16进制数,除此之外还可以使用Tk内置的颜色常量
# root_window["background"] = "#C9C9C9"
# # 添加文本内,设置字体的前景色和背景色，和字体类型、大小
# text = tk.Label(root_window, text="C语言中文网，欢迎您", bg="yellow", fg="red", font=('Times', 20, 'bold italic'))
# # 将文本内容放置在主窗口内
# text.pack()
# # 添加按钮，以及按钮的文本，并通过command 参数设置关闭窗口的功能
# button1 = tk.Button(root_window, text="关闭", command=root_window.quit)
# # 将按钮放置在主窗口内
# button1.pack(side="bottom")
#

# def button_click():
#     label.config(text="Button Clicked!")

# 全局变量
models = []
data_set = "AEEEM"
buttons_single = []
buttons_single_texts = ["naive_bayes", "svm"]
buttons_combine = []
buttons_combine_texts = ["naive_bayes", "svm"]
single = 0
combine = 0
single_model = 'naive_bayes'


# 提交按钮
def button_data_click():
    label_select.config(text="Button Clicked!")
    print(models)
    print('------')
    print(data_set)


# 组合模型预测的模型选择
def on_checkbox_click():
    models = []
    for checkbox_var, checkbox_text in zip(buttons_combine, buttons_combine_texts):
        if checkbox_var.get() == 1:
            models.append(checkbox_text)

    # if option1_var.get() == 1:
    #     models.append("naive_bayes")
    # if option2_var.get() == 1:
    #     models.append("svm")
    print("已选中的选项:", models)


# 数据集选择
def on_radio_button_click():
    global data_set
    data_set = radio_var.get()
    print("已选中的选项:", data_set)


# 单个模型选择
def radio_single_model():
    global single_model
    for i in range(len(buttons_single)):
        if buttons_single[i] == buttons_single_texts[i]:
            single_model = buttons_single_texts[i]
            print("选中的值为:", single_model)


# 选择组合模型预测
def show_models_combine():
    global combine
    global single
    if single == 1:
        container_single_model_show.pack_forget()
    container_combine_model_show.pack()
    if combine == 0:
        for i in range(len(buttons_combine_texts)):
            buttons_combine.append(check_combines)
            checkbox = tk.Checkbutton(container_combine_model_show, text=buttons_combine_texts[i],
                                      variable=check_combines,
                                      command=on_checkbox_click)
            checkbox.pack(side=tk.LEFT)
        combine = 1

    # option1_checkbox.pack(side=tk.LEFT)
    # option2_checkbox.pack(side=tk.LEFT)


# 选择单模型预测
def show_models_single():
    global combine
    global single
    if combine == 1:
        container_combine_model_show.pack_forget()
    container_single_model_show.pack()
    if single == 0:
        for i in range(len(buttons_single_texts)):
            buttons_single.append(buttons_single_texts[i])
            radiobutton = tk.Radiobutton(container_single_model_show, text=buttons_single_texts[i],
                                         variable=radio_single,
                                         value=buttons_single_texts[i], command=radio_single_model)
            radiobutton.pack(side=tk.LEFT)
        single = 1


root = tk.Tk()
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
button_models.pack(side=tk.LEFT)
button_models_single = tk.Button(container_model_show, text="组合模型预测", command=show_models_combine)
button_models_single.pack(side=tk.LEFT)
container_model_show.pack()
container_single_model_show = tk.Frame(container_model_select)
radio_single = tk.StringVar(value=buttons_single_texts[0])
container_single_model_show.pack()
container_combine_model_show = tk.Frame(container_model_select)
check_combines = tk.IntVar()
container_combine_model_show.pack()
container_model_select.pack()
button_data = tk.Button(root, text="确认选择", command=button_data_click)
button_data.pack(fill=tk.Y, pady=30)
label_select = tk.Label(root, text="")
label_select.pack(side=tk.BOTTOM)
root.mainloop()
