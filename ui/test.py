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


def button_data_click():
    label_select.config(text="Button Clicked!")


def on_checkbox_click():
    selected_options = []
    if option1_var.get() == 1:
        selected_options.append("选项1")
    if option2_var.get() == 1:
        selected_options.append("选项2")

    print("已选中的选项:", selected_options)


def on_radio_button_click():
    selected_option = radio_var.get()
    print("已选中的选项:", selected_option)


root = tk.Tk()
root.geometry('450x300')
label_title = tk.Label(root, text="缺陷预测模型")
label_title.pack(side='top')
# label = tk.Label(root, text="Hello, World!")
# label.pack()
# button = tk.Button(root, text="Click Me", command=button_click)
# button.pack()

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
option1_var = tk.IntVar()
option2_var = tk.IntVar()
option1_checkbox = tk.Checkbutton(container_model_select, text="naive_bayes", variable=option1_var,
                                  command=on_checkbox_click)
option1_checkbox.pack(side=tk.LEFT)
option2_checkbox = tk.Checkbutton(container_model_select, text="svm", variable=option2_var, command=on_checkbox_click)
option2_checkbox.pack(side=tk.LEFT)
container_model_select.pack()

button_data = tk.Button(root, text="确认选择", command=button_data_click)
button_data.pack(fill=tk.Y, pady=30)
label_select = tk.Label(root, text="")
label_select.pack(side=tk.BOTTOM)
root.mainloop()
