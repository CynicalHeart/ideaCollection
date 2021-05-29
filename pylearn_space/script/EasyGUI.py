# 简单GUI制作
import PySimpleGUI as sg


sg.theme('DarkBlue6')

# sg.preview_all_look_and_feel_themes()

layout = [[sg.Text('输入测试.')],
          [sg.InputText()],
          [sg.Submit(), sg.Cancel()]]

window = sg.Window('Window Title', layout)

event, values = window.read()
window.close()

text_input = values[0]
sg.popup('You entered', text_input)
