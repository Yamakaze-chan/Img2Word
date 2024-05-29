import flet as ft
import os
import datetime
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def History_Screen(page):
    # Main history panel
    history_panel = ft.ExpansionPanelList(
        expand_icon_color=ft.colors.AMBER,
        elevation=4,
        divider_color=ft.colors.AMBER,
    )

    # Child panel
    
    child_history_panel = ft.ExpansionPanel(
        # can_tap_header = True,
        header=ft.ListTile(
            title=ft.Text(f"Danh sách ảnh đã nhận ",
                          size=20, 
                          weight=ft.FontWeight.BOLD, 
                          theme_style=ft.TextThemeStyle.TITLE_LARGE, 
                          spans=[
                              ft.TextSpan(text="(tối đa hệ thống chỉ lưu 100 ảnh gần nhất)",
                                          style=ft.TextStyle(italic=True, size=15))])),    
        )
    history_panel.controls.append(child_history_panel)

    column_scroll_history_panel = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        height=500,
    )
    
    def open_file(e):
        if os.path.isfile(e.control.data):
            os.startfile(e.control.data)
    
    def open_folder_location(e):
        if os.path.isdir(e.control.data):
            os.startfile(e.control.data)

    def delete_image(e: ft.ControlEvent):
        # print(e.__dict__)
        # print(e.control.data)
        os.remove(e.control.data[0])
        column_scroll_history_panel.controls.remove(e.control.data[1])
        page.update()

    def from_image_to_docx(e: ft.ControlEvent):
        page.go('/file_to_doc/'+e.control.data)
    
    # get all recieve images
    for (dirpath, dirnames, filenames) in os.walk(resource_path(r'views\upload_img')):
        filenames.sort(key=lambda item: os.path.getctime(os.path.join(dirpath, item)), reverse=True)
        for file in filenames:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                delete_btn = ft.ElevatedButton(color='white', bgcolor=ft.colors.RED_600, text="Xóa", on_click=delete_image)
                container = ft.Container(
                        bgcolor='#282A36',
                        border_radius=ft.border_radius.all(15)
                    )
                container.content = ft.Row(
                                height= 500,
                                alignment = ft.MainAxisAlignment.CENTER,
                                vertical_alignment= ft.MainAxisAlignment.CENTER,
                                controls = [
                                    ft.Container(expand=2,margin=20, content=ft.Container(
                                        padding = 10,
                                        content=ft.Image(
                                                    src=os.path.join(dirpath, file),
                                                    fit=ft.ImageFit.CONTAIN,
                                                    repeat=ft.ImageRepeat.NO_REPEAT,
                                                )
                                    )),
                                    ft.Container(expand=2,margin=20, content=
                                                    ft.Container(
                                                    content=ft.Column(
                                                        alignment = ft.MainAxisAlignment.CENTER,
                                                        horizontal_alignment= ft.MainAxisAlignment.CENTER,
                                                        controls=[
                                                            ft.Text(
                                                                value=file,
                                                            ),
                                                            ft.Text(
                                                                value="Thời gian nhận file "+str(datetime.datetime.fromtimestamp(os.path.getctime(os.path.join(dirpath, file))).strftime("%H:%M:%S %d/%m/%Y")),
                                                            ),
                                                            ft.Row(
                                                                controls= [
                                                                    ft.ElevatedButton(data=os.path.join(dirpath, file), text="Mở file", on_click= open_file),
                                                                    ft.ElevatedButton(data=dirpath, text="Mở thư mục chứa file", on_click= open_folder_location),
                                                                    delete_btn,
                                                                ]
                                                            )
                                                        ]
                                                    )
                                                )),
                                    ft.Container(expand=1,
                                                 margin=20,
                                                 content=ft.FilledButton("Chuyển ảnh thành file docx",
                                                                         icon=ft.icons.DOCUMENT_SCANNER_OUTLINED,
                                                                         data=os.path.join(dirpath, file),
                                                                         on_click=from_image_to_docx)),
                                ]
                            )
                # child_history_main_panel_listview.controls.append(container)
                delete_btn.data = [os.path.join(dirpath, file),container]
                column_scroll_history_panel.controls.append(container)
        break
    child_history_panel.content = column_scroll_history_panel

    return ft.View(
        route = "/history",
        controls = [
            ft.AppBar(title=ft.Text("Lịch sử"), bgcolor=ft.colors.SURFACE_VARIANT),
            history_panel
        ],
    )