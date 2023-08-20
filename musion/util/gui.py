import flet as ft
from flet import FilePickerResultEvent
from flet.matplotlib_chart import MatplotlibChart
import librosa
import numpy as np
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt

from musion.util.pkg_util import task_names, get_task_instance, get_task_description

selected_files = ft.Text()
tabs: ft.Tabs = None
audio_file = None
tasks = {}

def draw_wav(audio_path):
    """
    display waveform of a given speech sample
    :param sample_name: speech sample name
    :param fs: sample frequency
    :return:
    """
    samples, sr = librosa.load(audio_path, sr=16000)
    samples = samples[6000:16000]

    print(len(samples), sr)
    time = np.arange(0, len(samples)) * (1.0 / sr)

    fig = plt.plot(time, samples)
    plt.title("语音信号时域波形")
    plt.xlabel("时长（秒）")
    plt.ylabel("振幅")
    # plt.savefig("your dir\语音信号时域波形图", dpi=600)
    plt.show()

    return fig

def add_audio_player(page: ft.Page, src):
    def volume_down(_):
        audio_file.volume -= 0.1
        audio_file.update()

    def volume_up(_):
        audio_file.volume += 0.1
        audio_file.update()

    # def balance_left(_):
    #     audio_file.balance -= 0.1
    #     audio_file.update()

    # def balance_right(_):
    #     audio_file.balance += 0.1
    #     audio_file.update()

    audio_file = ft.Audio(
        src=src,
        autoplay=False,
        volume=0.5,
        balance=0,
        # on_loaded=lambda _: print("Loaded"),
        # on_duration_changed=lambda e: print("Duration changed:", e.data),
        # on_position_changed=lambda e: print("Position changed:", e.data),
        # on_state_changed=lambda e: print("State changed:", e.data),
        # on_seek_complete=lambda _: print("Seek complete"),
    )
    page.overlay.append(audio_file)
    tabs.tabs[tabs.selected_index].content = ft.Column([
        ft.ElevatedButton("Play", on_click=lambda _: audio_file.play()),
        ft.ElevatedButton("Pause", on_click=lambda _: audio_file.pause()),
        ft.ElevatedButton("Resume", on_click=lambda _: audio_file.resume()),
        ft.ElevatedButton("Release", on_click=lambda _: audio_file.release()),
        # ft.ElevatedButton("Seek 2s", on_click=lambda _: audio_file.seek(2000)),
        ft.Row(
            [
                ft.ElevatedButton("Volume down", on_click=volume_down),
                ft.ElevatedButton("Volume up", on_click=volume_up),
            ]
        ),
        # ft.Row(
        #     [
        #         ft.ElevatedButton("Balance left", on_click=balance_left),
        #         ft.ElevatedButton("Balance right", on_click=balance_right),
        #     ]
        # ),
        # ft.ElevatedButton(
        #     "Get duration", on_click=lambda _: print("Duration:", audio_file.get_duration())
        # ),
        # ft.ElevatedButton(
        #     "Get current position",
        #     on_click=lambda _: print("Current position:", audio_file.get_duration()),
        # )
        ])
    
    page.update()

def processing_info_content(t: ft.Tabs):
    pr = ft.ProgressRing(width=16, height=16, stroke_width = 2)
    pb = ft.ProgressBar(width=400, value=0)
    pt = ft.Text()
    t.tabs[t.selected_index].content = ft.Column(controls=[
        ft.Text('Processing...'),
        pr,
        pb],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    t.update()

    return pb, pt

def add_tabs(page: ft.Page):
    global tabs

    def select_tab(e: ft.ControlEvent):
        # if not selected_files.value:
        #     return
        # pb, pt = processing_info_content(tabs)
        # page.update()
        # task_name = task_names[tabs.selected_index]
        # if task_name not in tasks.keys():
        #     musion_task = get_task_instance(task_name)
        #     musion_task.observers += [pb]
        #     tasks[task_name] = {}
        #     tasks[task_name]['object'] = musion_task
        # tasks[task_name]['res'] = tasks[task_name]['object'](audio_path=selected_files.value)
        # tabs.tabs[tabs.selected_index].content = ft.Text(str(tasks[task_name]['res']))
        page.update()

    tab_controls = []
    for task in task_names:
        tab_controls.append(ft.Tab(text=task, content=ft.Column([ft.Text(get_task_description(task)),
                                                              ft.Text('Pick a file to analyze.')],
                                                              alignment=ft.MainAxisAlignment.CENTER,
                                                              horizontal_alignment=ft.CrossAxisAlignment.CENTER)))
    tabs = ft.Tabs(
        # selected_index=0,
        animation_duration=300,
        scrollable=False,
        tabs=tab_controls,
        expand=1,
        # on_change=select_tab
    )

    page.add(tabs, selected_files)

def execute_musion_task(task_name):
    pb, pt = processing_info_content(tabs)
    if task_name not in tasks.keys():
        musion_task = get_task_instance(task_name)
        musion_task.observers += [pb]
        tasks[task_name] = {}
        tasks[task_name]['object'] = musion_task
    tasks[task_name]['res'] = tasks[task_name]['object'](audio_path=selected_files.value)

def analyze_click(e):
    if not selected_files.value:
        # TODO Alert
        return

    task_name = task_names[tabs.selected_index]
    execute_musion_task(task_name)
    tabs.tabs[tabs.selected_index].content = ft.Column(
        [ft.ElevatedButton("Save", on_click=analyze_click),
        ft.Text(str(tasks[task_name]['res']))
        # MatplotlibChart(draw_wav(selected_files.value), expand=True)
        ])
    tabs.update()

def clear_tabs():
    if tabs.tabs:
        for t in tabs.tabs:
            t.content = ft.Row([ft.ElevatedButton("Analyze", on_click=analyze_click)],
                               alignment=ft.MainAxisAlignment.CENTER,
                               vertical_alignment=ft.CrossAxisAlignment.CENTER)
        tabs.update()

def main(page: ft.Page):
    page.title = "mUSIon"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # Pick files dialog
    def pick_files_result(e: FilePickerResultEvent):
        selected_files.value = e.files[0].path if e.files else "Cancelled!"

        if selected_files.value == "Cancelled!":
            return

        clear_tabs()

        # add_audio_player(page, selected_files.value)

        selected_files.update()

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)

    # hide all dialogs in overlay
    page.overlay.extend([pick_files_dialog])

    page.add(
        ft.Row(
            [
                ft.ElevatedButton(
                    "Pick file",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: pick_files_dialog.pick_files(),
                ),
                selected_files,
            ],
        ),
        # ft.Row(
        #     [
        #         ft.IconButton(ft.icons.REMOVE, on_click=minus_click),
        #         txt_number,
        #         ft.IconButton(ft.icons.ADD, on_click=plus_click),
        #     ],
        #     alignment=ft.MainAxisAlignment.CENTER,
        # )
    )

    add_tabs(page)

def main_gui():
    ft.app(target=main,
        #    view=ft.AppView.WEB_BROWSER
        )

if __name__ == '__main__':
    main_gui()