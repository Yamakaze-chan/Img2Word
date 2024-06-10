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

    # Child Image panel
    child_history_image_panel = ft.ExpansionPanel(
        can_tap_header=True,
        expand=True,
        expand_loose = True,
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
    history_panel.controls.append(child_history_image_panel)

    column_scroll_history_image_panel = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        height=450,
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
        column_scroll_history_image_panel.controls.remove(e.control.data[1])
        page.update()

    def from_image_to_docx(e: ft.ControlEvent):
        page.go('/file_to_doc/'+e.control.data)
    
    # get all recieve images
    for (dirpath_img, dirnames_img, filenames_img) in os.walk(resource_path(r'upload_img')):
        filenames_img.sort(key=lambda item: os.path.getctime(os.path.join(dirpath_img, item)), reverse=True)
        for file in filenames_img:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                delete_img_btn = ft.ElevatedButton(color='white', bgcolor=ft.colors.RED_600, text="Xóa", on_click=delete_image)
                container_img = ft.Container(
                        bgcolor='#282A36',
                        border_radius=ft.border_radius.all(15)
                    )
                container_img.content = ft.Row(
                                height= 500,
                                alignment = ft.MainAxisAlignment.CENTER,
                                vertical_alignment= ft.MainAxisAlignment.CENTER,
                                controls = [
                                    ft.Container(expand=2,margin=20, content=ft.Container(
                                        padding = 10,
                                        content=ft.Image(
                                                    src=os.path.join(dirpath_img, file),
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
                                                                value="Thời gian nhận file "+str(datetime.datetime.fromtimestamp(os.path.getctime(os.path.join(dirpath_img, file))).strftime("%H:%M:%S %d/%m/%Y")),
                                                            ),
                                                            ft.Row(
                                                                controls= [
                                                                    ft.ElevatedButton(data=os.path.join(dirpath_img, file), text="Mở file", on_click= open_file),
                                                                    ft.ElevatedButton(data=dirpath_img, text="Mở thư mục chứa file", on_click= open_folder_location),
                                                                    delete_img_btn,
                                                                ]
                                                            )
                                                        ]
                                                    )
                                                )),
                                    ft.Container(expand=1,
                                                 margin=20,
                                                 content=ft.FilledButton("Chuyển ảnh thành file docx",
                                                                         icon=ft.icons.DOCUMENT_SCANNER_OUTLINED,
                                                                         data=os.path.join(dirpath_img, file),
                                                                         on_click=from_image_to_docx)),
                                ]
                            )
                # child_history_main_panel_listview.controls.append(container)
                delete_img_btn.data = [os.path.join(dirpath_img, file),container_img]
                column_scroll_history_image_panel.controls.append(container_img)
        break
    child_history_image_panel.content = column_scroll_history_image_panel

    # Child doc panel
    child_history_doc_panel = ft.ExpansionPanel(
        can_tap_header=True,
        expand=True,
        expand_loose = True,
        # can_tap_header = True,
        header=ft.ListTile(
            title=ft.Text(f"Danh sách file Word đã chuyển ",
                          size=20, 
                          weight=ft.FontWeight.BOLD, 
                          theme_style=ft.TextThemeStyle.TITLE_LARGE, 
                          spans=[
                              ft.TextSpan(text="(tối đa hệ thống chỉ lưu 100 file gần nhất)",
                                          style=ft.TextStyle(italic=True, size=15))])),    
        )
    history_panel.controls.append(child_history_doc_panel)

    column_scroll_history_doc_panel = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        height=450,
    )

    def delete_doc(e: ft.ControlEvent):
        # print(e.__dict__)
        # print(e.control.data)
        os.remove(e.control.data[0])
        column_scroll_history_doc_panel.controls.remove(e.control.data[1])
        page.update()

    # get all recieve images
    for (dirpath_doc, dirnames_doc, filenames_doc) in os.walk(resource_path(r'result')):
        filenames_doc.sort(key=lambda item: os.path.getctime(os.path.join(dirpath_doc, item)), reverse=True)
        for file in filenames_doc:
            if file.lower().endswith(('.docx', '.doc')):
                delete_doc_btn = ft.ElevatedButton(color='white', bgcolor=ft.colors.RED_600, text="Xóa", on_click=delete_doc)
                container_doc = ft.Container(
                        bgcolor='#282A36',
                        border_radius=ft.border_radius.all(15)
                    )
                container_doc.content = ft.Row(
                                alignment = ft.MainAxisAlignment.CENTER,
                                vertical_alignment= ft.MainAxisAlignment.CENTER,
                                controls = [
                                    ft.Container(expand=1,margin=20, content=ft.Container(
                                        padding = 10,
                                        content=ft.Icon(name=ft.icons.TEXT_SNIPPET, color=ft.colors.BLUE)
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
                                                                value="Thời gian nhận file "+str(datetime.datetime.fromtimestamp(os.path.getctime(os.path.join(dirpath_doc, file))).strftime("%H:%M:%S %d/%m/%Y")),
                                                            ),
                                                            ft.Row(
                                                                controls= [
                                                                    ft.ElevatedButton(data=os.path.join(dirpath_doc, file), text="Mở file", on_click= open_file),
                                                                    ft.ElevatedButton(data=dirpath_doc, text="Mở thư mục chứa file", on_click= open_folder_location),
                                                                    delete_doc_btn,
                                                                ]
                                                            )
                                                        ]
                                                    )
                                                )),
                                ]
                            )
                # child_history_main_panel_listview.controls.append(container)
                delete_doc_btn.data = [os.path.join(dirpath_doc, file),container_doc]
                column_scroll_history_doc_panel.controls.append(container_doc)
        break
    child_history_doc_panel.content = column_scroll_history_doc_panel

    return ft.View(
        route = "/history",
        controls = [
            ft.AppBar(title=ft.Text(spans=[
                ft.TextSpan(text="Lịch Sử"),
                ft.TextSpan(text=" (Hệ thống chỉ lưu trong thư mục mặc định của phần mềm)", style=ft.TextStyle(italic=True, size=15))
            ]), bgcolor=ft.colors.SURFACE_VARIANT),
            history_panel
        ],
    )